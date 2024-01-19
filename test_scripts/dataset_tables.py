"""Dataset Info Tables Test Program

This script generates tables describing compiled information.
In the future, this test should lead to an application that generates .tex tables to be used in the dataset publication article.
"""
import pandas as pd
import iara.description


def main(show_sample_dataset = False):
    """Main function for the dataset info tables."""

    os_ship_merged = []
    for sub in iara.description.Subdataset:
        if sub.value > iara.description.Subdataset.os_ship.value:
            continue

        df = sub.to_dataframe(only_sample=show_sample_dataset)
        part = df.groupby(['TYPE','DETAILED TYPE']).size().reset_index(name=str(sub))

        if not isinstance(os_ship_merged, pd.DataFrame):
            os_ship_merged = part
        else:
            os_ship_merged = pd.merge(os_ship_merged, part, on=['TYPE','DETAILED TYPE'], how='outer')

    os_ship_merged = os_ship_merged.fillna(0)
    os_ship_merged = os_ship_merged.sort_values(['TYPE','DETAILED TYPE'])

    print('------------------------ os_ship_merged ----------------------------------------')
    print(os_ship_merged)


    os_bg = iara.description.Subdataset.E.to_dataframe(only_sample=show_sample_dataset)
    os_bg_merged = os_bg.groupby(['Rain state', 'Sea state']).size().reset_index(name=str(sub))

    order = {str(rain_enum): rain_enum.value for rain_enum in iara.description.Rain}
    os_bg_merged['Order'] = os_bg_merged['Rain state'].map(order)
    os_bg_merged = os_bg_merged.sort_values('Order').drop('Order', axis=1).reset_index(drop=True)

    print('\n------------------------- os_bg_merged -----------------------------------------')
    print(os_bg_merged)

    
if __name__ == "__main__":
    main()