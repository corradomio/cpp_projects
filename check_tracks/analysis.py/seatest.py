import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_theme(style="darkgrid")

print(sns.__version__)

# Load an example dataset with long-form data
fmri = sns.load_dataset("fmri")
print(fmri.head())

# Plot the responses for different events and regions
sns.lineplot(x="timepoint", y="signal",
             hue="region", style="event",
             data=fmri)
plt.show()

import pandas as pd

ds = pd.DataFrame()
se = pd.Series()
ix = pd.Index()