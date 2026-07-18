import os

from ros_base.launch.base_launcher import BaseLauncher
from ros_base.utils.args_debug import add_debug_mode

if __name__ == "__main__":
    parser = add_debug_mode(listen_port=9997)
    default_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deploy_cfg.yaml")
    parser.add_argument("config", nargs="?", default=default_cfg, help="Path to YAML config file")
    parser.add_argument("--enable", nargs="+", default=[], help="List of node names to enable explicitly")
    parser.add_argument("--disable", nargs="+", default=[], help="List of node names to disable explicitly")
    args = parser.parse_args()

    launcher = BaseLauncher(args.config, enable_nodes=args.enable, disable_nodes=args.disable)
    launcher.launch()
