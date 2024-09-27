import argparse
import param_estimation


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_dir", default="data/data_cond/", type=str)
    argparser.add_argument("--save_path", default="cst_model/params/model_params.pkl", type=str)
    argparser.add_argument("--tick_size", default=100, type=int)
    argparser.add_argument("--num_ticks", default=500, type=int)
    argparser.add_argument("--recompute_existing", action="store_true")

    args = argparser.parse_args()

    print("[*] Estimating params...")
    aggr_params = param_estimation.estimate_from_data_files(
        args.data_dir,
        save_path=args.save_path,
        tick_size=args.tick_size,
        num_ticks=args.num_ticks,
        recompute_existing=args.recompute_existing,
    )
    print("[*] Done")
