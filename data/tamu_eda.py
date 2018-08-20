import numpy as np
import pandas as pd

%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import json

tamu_df = pd.read_json('tamu_database.json')

# Rearranging and cleaning tamu_df
tamu_df_rearranged = tamu_df.copy()
tamu_df_rearranged.drop(columns='university_name', inplace=True)
tamu_df_rearranged.rename(index=str, columns={"faculty_names": "faculty_info"}, inplace=True)
tamu_df_rearranged["faculty_name"] = tamu_df_rearranged.index
tamu_df_rearranged.reset_index(drop=True, inplace=True)

# Expanding faculty_info column
faculty_info_df = tamu_df_rearranged["faculty_info"].apply(pd.Series)
tamu_df_final = pd.concat([tamu_df_rearranged, faculty_info_df], axis=1).drop("faculty_info", axis=1)
