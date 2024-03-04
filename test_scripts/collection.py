"""
Dataset Info Tables Test Program

This script generates tables describing compiled information.
In the future, this test should lead to an application that generates .tex tables
    to be used in the dataset publication article.
"""
import os
import pandas as pd

import iara.records
from iara.default import DEFAULT_DIRECTORIES


def main(show_sample_dataset = False):
    """Main function for the dataset info tables."""

    os.makedirs(DEFAULT_DIRECTORIES.tables_dir, exist_ok=True)

    os_ship_merged = []
    for sub in iara.records.Collection:
        if sub.value > iara.records.Collection.OS_SHIP.value:
            continue

        df = sub.to_df(only_sample=show_sample_dataset)
        part = df.groupby(['TYPE','DETAILED TYPE']).size().reset_index(name=str(sub))

        if not isinstance(os_ship_merged, pd.DataFrame):
            os_ship_merged = part
        else:
            os_ship_merged = pd.merge(os_ship_merged, part,
                                      on=['TYPE','DETAILED TYPE'],how='outer')

    os_ship_merged = os_ship_merged.fillna(0)
    os_ship_merged = os_ship_merged.sort_values(['TYPE','DETAILED TYPE'])

    keeped = os_ship_merged[os_ship_merged['Total']>=20]

    filtered = os_ship_merged[os_ship_merged['Total']<20]
    filtered = filtered.groupby('TYPE').sum()
    filtered['DETAILED TYPE'] = 'Others'
    filtered.reset_index(inplace=True)

    os_ship_detailed_type = pd.concat([keeped, filtered])
    os_ship_detailed_type.sort_values(by='DETAILED TYPE', inplace=True)
    os_ship_detailed_type.sort_values(by='TYPE', inplace=True)
    os_ship_detailed_type.reset_index(drop=True, inplace=True)
    os_ship_detailed_type.loc['Total'] = os_ship_detailed_type.sum()
    os_ship_detailed_type.loc[os_ship_detailed_type.index[-1], 'TYPE'] = 'Total'
    os_ship_detailed_type.loc[os_ship_detailed_type.index[-1], 'DETAILED TYPE'] = 'Total'
    os_ship_detailed_type[os_ship_detailed_type.columns[2:]] = \
            os_ship_detailed_type[os_ship_detailed_type.columns[2:]].astype(int)
    os_ship_detailed_type.to_latex(
            os.path.join(DEFAULT_DIRECTORIES.tables_dir, 'os_ship_detailed_type.tex'), index=False)

    print('------------------------ os_ship_detailed_type ----------------------------------------')
    print(os_ship_detailed_type)


    os_ship_type = os_ship_merged.groupby('TYPE').sum()
    os_ship_type = os_ship_type.drop('DETAILED TYPE', axis=1)
    os_ship_type.loc['Total'] = os_ship_type.sum()
    os_ship_type[os_ship_type.columns] = \
            os_ship_type[os_ship_type.columns].astype(int)
    os_ship_type.to_latex(
            os.path.join(DEFAULT_DIRECTORIES.tables_dir, 'os_ship_type.tex'), index=True)

    print('------------------------ os_ship_type ----------------------------------------')
    print(os_ship_type)


    # os_bg = iara.records.Collection.E.to_df(only_sample=show_sample_dataset)
    # os_bg_merged = os_bg.groupby(['Rain state', 'Sea state']).size().reset_index(name='Qtd')

    # order = {str(rain_enum): rain_enum.value for rain_enum in iara.records.Rain}
    # os_bg_merged['Order'] = os_bg_merged['Rain state'].map(order)
    # os_bg_merged = os_bg_merged.sort_values('Order').drop('Order', axis=1).reset_index(drop=True)

    # print('\n------------------------- os_bg_merged -----------------------------------------')
    # print(os_bg_merged)

if __name__ == "__main__":
    main()
