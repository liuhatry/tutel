r"""
Utility in Megatron
"""
def add_moe_args(parser):
    group = parser.add_argument_group(title='moe')

    group.add_argument("--num-experts", type=int, default=None)
    group.add_argument("--top-k", type=int, default=2)
    group.add_argument('--expert-hidden-size', type=int, default=None,
                       help='Expert hidden size.')

    return parser
