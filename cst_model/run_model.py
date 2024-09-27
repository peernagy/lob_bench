import sys
import jax
import jax.numpy as jnp
import glob
import argparse
import pathlib
from tqdm import tqdm
from jaxlob.gymnax_exchange.jaxob.jorderbook import OrderBook
from jaxlob.gymnax_exchange.jaxob.jaxob_config import Configuration as JaxLOBConfig
import cst
import lobster_conversion
from param_estimation import load_params

# FIXME: this is a hack to import from parent dir
# add parent dir to path
sys.path.append("..")
from lob_bench import data_loading


START_OID = 100
SIM_CONFIG = JaxLOBConfig()


def generate_data(
    gen_seq_len: int,
    n_lvls: int,
    params_file: str,
    data_dir: str,
    save_dir: str,
    seed: int = 0,
):
    # create save_dir if it does not exist
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    aggr_params_dict = load_params(params_file)
    params, lo_lambda, co_theta = cst.init_params(aggr_params_dict)

    orderbook_files = sorted(glob.glob(data_dir + "/*orderbook*.csv"))
    for obf in tqdm(orderbook_files):
        # load book (conditioning data)
        lobster_book = data_loading.load_book_df(obf)
        lobster_msgs = data_loading.load_message_df(obf.replace("orderbook", "message"))
        # take the last row available as initial book
        init_book = cst.init_book(
            lobster_book.iloc[-1].values,
            params,
            float(lobster_msgs.iloc[-1, 0]),
        )
        l2_init, init_time = cst.get_l2_book(init_book, params, n_lvls)
        base_rates = cst.get_event_base_rates(
            params,
            cancel_rates=co_theta,
            lo_rates=lo_lambda,
        )
        rng = jax.random.PRNGKey(seed)

        sim = OrderBook(SIM_CONFIG)
        sim_state = sim.reset(l2_init.flatten())

        final_carry, (l2_books, lobster_msgs) = jax.lax.scan(
            lobster_conversion.make_step_book_scannable(10),
            (sim, sim_state, init_book, base_rates, params, START_OID, rng),
            length=gen_seq_len,
        )

        # save generated book data to csv
        cst.l2_book_to_pandas(l2_books).to_csv(
            save_dir + "/" + obf.rsplit("/", maxsplit=1)[-1] + f"_gen_id_{seed}.csv",
            index=False,
            header=False,
        )
        # save generated message data to csv
        lobster_msgs.to_pandas().to_csv(
            save_dir + "/"
            + obf.rsplit("/", maxsplit=1)[-1].replace("orderbook", "message")
            + f"_gen_id_{seed}.csv",
            index=False,
            float_format='%.9f',
            header=False,
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--gen_seq_len", type=int)
    argparser.add_argument("--n_levels", type=int, default=10)
    argparser.add_argument("--params_file", type=str, default="cst_model/params/model_params.pkl")
    argparser.add_argument("--data_dir", type=str, default="data/data_cond/")
    argparser.add_argument("--save_dir", type=str, default="data/data_gen_bench/")
    args = argparser.parse_args()

    generate_data(
        gen_seq_len=args.gen_seq_len,
        n_lvls=args.n_levels,
        params_file=args.params_file,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
    )
