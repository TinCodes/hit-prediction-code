#!/usr/bin/env python3

from dataloaders import MsdBbLoader


def main():
    loader = MsdBbLoader(
        hits_file_path=
        './storage/msd_bb_matches.csv',
        non_hits_file_path=
        './storage/msd_bb_non_matches.csv',
        features_path='./tools',
        non_hits_per_hit=1,
        features=['hl'],
        label='weeks',
    )

    print(loader.load()[0][0].shape, loader.load()[1].shape)
    print(loader.configuration())
    print(loader.load()[0][0].columns)


if __name__ == '__main__':
    main()
