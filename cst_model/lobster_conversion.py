from functools import partial
import pandas as pd
import jax
import jax.numpy as jnp
import chex
import jaxlob
import jaxlob.gymnax_exchange.jaxob.JaxOrderBookArrays as job
from jaxlob.gymnax_exchange.jaxob.jorderbook import OrderBook, LobState
from jaxlob.gymnax_exchange.jaxob.jaxob_config import Configuration as JaxLOBConfig
import cst
from cst import CSTParams


START_OID = 100
SIM_CONFIG = JaxLOBConfig()


@chex.dataclass
class LobsterMessage(cst.Message):
    oid: jax.Array

    def to_pandas(self):
        return pd.DataFrame({
            "time": self.time,
            "event_type": self.event_type,
            "oid": self.oid,
            "size": self.size,
            "price": self.price,
            "direction": self.direction,
        })


def msg_to_jaxlob(msg: cst.Message, oid: jax.Array) -> jax.Array:
    return jnp.array([
        msg.event_type,
        msg.direction,
        jnp.abs(msg.size),
        msg.price,
        0, # trader ID (not used)
        oid, # order ID
        msg.time, # converts whole seconds to int
        msg.time % 1 * 1e9, # converts fractional seconds to nanoseconds
    ], dtype=jnp.int32)


def update_oid(
    msg: jax.Array,
    rng: jax.Array,
    sim: OrderBook,
    sim_state: LobState,
) -> jax.Array:
    def _leave_oid(msg, rng, *args):
        return msg, rng

    # 3 -> random from active at price
    def _get_random_cancel_msg(msg, rng, sim, sim_state):
        side = msg[1]
        side_array = jax.lax.cond(
            side == 1,
            lambda a, b: b,
            lambda a, b: a,
            sim_state.asks, sim_state.bids
        )
        rng, _rng = jax.random.split(rng)
        msg_dict = {
            "quantity": msg[2],
            "price": msg[3],
        }
        # TODO: find out why this returns -1 and is not working
        idx = job.get_random_id_match(SIM_CONFIG, _rng, side_array, msg_dict)
        cancelled_oid = side_array[idx, 2]
        jax.debug.print('cancelled_oid {}', cancelled_oid)
        msg = msg.at[5].set(cancelled_oid)
        return msg, rng

    # 4 -> get oid from active at price
    def _get_oid_from_active(msg, rng, sim, sim_state):
        # oid = sim.get_next_executable_order(sim_state, msg[1])[3]
        side = msg[1]
        oid = jax.lax.cond(
            side == 1,
            _get_top_bid_order_oid,
            _get_top_ask_order_oid,
            sim_state
        )
        jax.debug.print('active_oid {}', oid)
        return msg.at[5].set(oid), rng

    def _get_top_bid_order_oid(sim_state):
        idx = job._get_top_bid_order_idx(SIM_CONFIG, sim_state.bids).squeeze()
        return sim_state.bids[idx, 2]

    def _get_top_ask_order_oid(sim_state):
        idx = job._get_top_ask_order_idx(SIM_CONFIG, sim_state.asks).squeeze()
        return sim_state.asks[idx, 2]

    msg, rng = jax.lax.switch(
        # 0 -> leave, 1 -> random cancel, 2 -> get executable oid from active
        (msg[0] == 3) + 2*(msg[0] == 4),
        (_leave_oid, _get_random_cancel_msg, _get_oid_from_active),
        msg, rng, sim, sim_state
    )
    return msg, rng


def make_step_book_scannable(n_levels: int):
    def _sim_step_scannable(carry, _) -> jax.Array:
        sim, sim_state, book, base_rates, params, new_oid, rng = carry

        book, message, rng = cst.step_book(book, base_rates, params, rng)
        l2_book, time = cst.get_l2_book(book, params, n_levels)
        msg_jaxlob = msg_to_jaxlob(message, new_oid)
        msg_jaxlob, rng = update_oid(msg_jaxlob, rng, sim, sim_state)
        sim_state = sim.process_order_array(sim_state, msg_jaxlob)
        # convert message without OID to lobster message with OID
        lobster_msg = LobsterMessage(
            time=message.time,
            event_type=message.event_type,
            oid=msg_jaxlob[5],
            size=message.size,
            price=message.price,
            direction=message.direction,
        )
        carry = (sim, sim_state, book, base_rates, params, new_oid + 1, rng)
        return carry, (l2_book.flatten(), lobster_msg)

    return _sim_step_scannable

if __name__ == "__main__":
    # with jax.disable_jit():

    params = CSTParams()
    book = cst.init_book(
        jnp.array([100_0100, 1, 100_0000, 2, 101_1000, 3, 98_0000, 4, 0, 0, 0, 0]),
        params
    )
    print("INITIAL BOOK")
    print(book)
    l2_init, init_time = cst.get_l2_book(book, params, 5)
    print(l2_init, init_time)

    co_theta = jnp.ones(params.num_ticks) * 0.1
    base_rates = cst.get_event_base_rates(params, co_theta)
    print("base_rates", base_rates)

    rng = jax.random.PRNGKey(0)

    # c, (msgs, l2_books) = jax.lax.scan(
    #     make_step_book_scannable(5),
    #     (book, base_rates, params, rng),
    #     length=100,
    # )
    # print(msgs)
    # print(l2_books)

    # sim_config = JaxLOBConfig()
    sim = OrderBook(SIM_CONFIG)
    sim_state = sim.reset(l2_init.flatten())
    # print(sim_state)

    # TODO: turn into a scan
    # for i in range(100):
    #     book, message, rng = cst.step_book(book, base_rates, params, rng)
    #     print(message)
    #     print("BOOK AFTER STEP", i)
    #     print(cst.get_l2_book(book, params, 5))
    #     # print(book)

    #     # start OIDs for LOs from 100 upwards
    #     msg_jaxlob = msg_to_jaxlob(message, jnp.array(i) + 100)
    #     msg_jaxlob, rng = update_oid(msg_jaxlob, rng, sim, sim_state, sim_config)
    #     print("msg_jaxlob", msg_jaxlob)
    #     sim_state = sim.process_order_array(sim_state, msg_jaxlob)
    #     print("JAX LOB")
    #     print(sim.get_L2_state(sim_state, n_levels=10).reshape(-1, 4))
    #     print()
    #     lobster_msg = LobsterMessage(
    #         time=message.time,
    #         event_type=message.event_type,
    #         oid=msg_jaxlob[5],
    #         size=message.size,
    #         price=message.price,
    #         direction=message.direction,
    #     )

    final_carry, (l2_books, lobster_msgs) = jax.lax.scan(
        make_step_book_scannable(10),
        (sim, sim_state, book, base_rates, params, START_OID, rng),
        length=100,
    )

    print(cst.l2_book_to_pandas(l2_books))
    # print(lobster_msgs)

    print(lobster_msgs.to_pandas())
