import os, sys, json
sys.path.insert(0, ".")
import torch.multiprocessing as mp


def worker(rank, world_size, strategy, port, result_path):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    from profile_run import build_arg_parser, run
    argv = ["--strategy", strategy, "--steps", "5", "--batch-size", "4", "--block-size", "16",
            "--d-model", "16", "--n-layer", "2", "--n-head", "2", "--backend", "gloo",
            "--log-every", "1000", "--out-dir", "/tmp/audit_" + strategy]
    args = build_arg_parser().parse_args(argv)
    summary, pr = run(args)
    if rank == 0:
        names = [o.name for o in pr.comm_ops] + [o.name for o in pr.compute_ops]
        suspicious = [n for n in names if "broadcast" in n.lower()]
        with open(result_path, "w") as f:
            json.dump({"suspicious": suspicious, "comm_names": [o.name for o in pr.comm_ops]}, f)


if __name__ == "__main__":
    strategy = sys.argv[1]
    nprocs = int(sys.argv[2])
    port = int(sys.argv[3])
    result_path = f"/tmp/audit_{strategy}_result.json"
    mp.spawn(worker, args=(nprocs, strategy, port, result_path), nprocs=nprocs, join=True)
    print(open(result_path).read())
