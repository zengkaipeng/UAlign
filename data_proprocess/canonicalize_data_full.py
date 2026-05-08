from canonicalize_data_csv import canonicalize_file, build_parser


def main():
    parser = build_parser(
        'Canonicalize USPTO-FULL CSV data',
        include_class_mode=False,
        default_num_procs=4,
    )
    args = parser.parse_args()
    canonicalize_file(
        args.filename,
        class_mode='minus_one',
        num_procs=args.num_procs,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
