<style>
.icon:hover {
  opacity: 0.7;
}
.icon { 
    overflow: hidden;
    filter: grayscale(100%);
}
</style>
<a href="https://www.linkedin.com/in/joel-montavon-704808a/" target="_blank"><img class="icon" width="60" height="60" src="https://content.linkedin.com/content/dam/me/brand/en-us/brand-home/logos/In-Blue-Logo.png.original.png" style="position: absolute; right: 80px; top: 10px;"></img></a>
<a href="https://github.com/joelmontavon/pdc-python" target="_blank"><img class="icon" width="80" height="80" src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" style="position: absolute; right: 0px; top: 0px;"></img></a>

<h1>Calculating the Proportion of Days Covered using Python</h1>

The proportion of days covered (PDC) is a method for calculating medication adherence. It involves identifying days covered based upon the date of service and days supply using prescription claims data. The methodology adjusts the start date for overlapping fills of the same medication. This makes sense because patients often come into the pharmacy to pickup their drugs a few days early so they do not run out of supply.

PDC is a more conservative estimate when patient switches between medications in the same class or concurrently uses more than one medication in a class. For most drug classes, a PDC ≥ 80% is considered adherent.

To make this easier in Python, I created a general-purpose class called a DateIndexedArray. This class allows me to represent a prescription claim as a date of fill and the days supply as an array. The elements of the array can be accessed either via the index or a date. And, I've created some of functions that help with some common tasks.

To calculate the PDC, we need to identify the days covered by each medication. For claims involving the same drug with overlapping days supply, we assume that the patient will finish his/her current fill before starting the refill. This means that we need to adjust for overlapping days supply which can be accomplished with the right shift operator (i.e., >>). 

For overlapping fills of different drug, we assume that the patient will start his/her new medication right away. So, we just need to sum for each index in the array which can be accomplished with the addition operator (i.e., +).


```python
import datetime as dt
import numpy as np
from functools import reduce

class DateIndexedArray():
    
    def __init__(self, epoch, size=365, val=1, type=float):
        self.epoch = epoch
        self.array = np.ndarray(size, type)
        self.array.fill(val)
        
    def __getitem__(self, key):
        if isinstance(key, dt.datetime):
            key = (key - self.epoch).days
        return self.array.__getitem__(key)
    
    def __str__(self):
        return self.epoch.strftime("%m/%d/%Y") + ': ' + self.array.__str__()
    
    def reindex(self, epoch, size):
        z = DateIndexedArray(epoch, size, 0)
        offset = (self.epoch - epoch).days
        z.array[offset:offset + self.array.size] += self.array
        return z
    
    def extend(self, y):
        firstDate = min([self.epoch, y.epoch])
        lastDate = max([self.epoch + dt.timedelta(days = self.array.size - 1), y.epoch + dt.timedelta(days = y.array.size - 1)])
        return self.reindex(firstDate, (lastDate - firstDate).days + 1)
    
    def trim(self, start, end):
        epoch = max([self.epoch, start])
        offset = max([(epoch - self.epoch).days, 0])
        self.array = self.array[offset:offset + (end - epoch).days + 1]
        self.epoch = epoch
        return self + DateIndexedArray(self.epoch, (end - epoch).days + 1, 0)
    
    def __add__(self, y):
        z = self.extend(y)
        z.array[(y.epoch - z.epoch).days:(y.epoch - z.epoch).days + y.array.size] += y.array
        return z
    
    def __rshift__(self, y):
        a = self.extend(y)
        b = y.reindex(a.epoch, a.array.size)
        zero = [j for j, val in enumerate(a.array) if val == 0 or np.isnan(val)]
        nonzero = [i for i, val in enumerate(b.array) if val != 0 and not np.isnan(val)]
        
        j = 0
        for i in nonzero:
            while (True):
                if j > (len(zero) - 1):
                    a.array = np.append(a.array, b[i])
                    j += 1
                    break
                elif zero[j] >= i:
                    a.array[zero[j]] = b[i]
                    j += 1
                    break
                else: 
                    j += 1

        return a
    
    def __lshift__(self, y):
        return y >> self
```

You can play around with this class a bit. Notice the difference in the results when performing addition versus a right shift.


```python
rx_clm1 = DateIndexedArray(dt.datetime(2022, 1, 1, 0, 0), 10)
print("rx_clm1: " + str(rx_clm1))
rx_clm2 = DateIndexedArray(dt.datetime(2022, 1, 7, 0, 0), 10)
print("rx_clm2: " + str(rx_clm2))
print("rx_clm1 + rx_clm2: " + str(rx_clm1 + rx_clm2))
print("rx_clm1 >> rx_clm2: " + str(rx_clm1 >> rx_clm2))
```

    rx_clm1: 01/01/2022: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
    rx_clm2: 01/07/2022: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
    rx_clm1 + rx_clm2: 01/01/2022: [1. 1. 1. 1. 1. 1. 2. 2. 2. 2. 1. 1. 1. 1. 1. 1.]
    rx_clm1 >> rx_clm2: 01/01/2022: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
    

I wanted to keep the DateIndexedArray class general-purpose so created some functions that are specific to calculating PDC. These functions allow me to things like identify the treatment period and calculate the days in the treatment period as well as PDC.


```python
def get_days_covered(arr, same_drug=True):
    if same_drug:
        return reduce(lambda x, y: x >> y, arr.copy())
    else:
        return reduce(lambda x, y: x + y, arr.copy())

def get_tx_period(arr, start, end):
    return arr.trim(start, end)

def get_tot_days_in_tx_period(arr):
    return arr.array.size

def get_tot_days_covered(arr, min_drugs):
    return np.count_nonzero(arr.array >= min_drugs)
```

Let's take a look at all of this in action. First, we need to create some sample prescription claims data.


```python
rx_clms = [
    {'pt_id': 'SAMEDRUG', 'drug_name': 'LISINOPRIL 10 MG TABS', 'drug': 'LISINOPRIL', 'date_of_service': '2021-01-01', 'days_sup': 90},
    {'pt_id': 'SAMEDRUG', 'drug_name': 'LISINOPRIL 10 MG TABS', 'drug': 'LISINOPRIL', 'date_of_service': '2022-01-01', 'days_sup': 90},
    {'pt_id': 'SAMEDRUG', 'drug_name': 'LISINOPRIL 10 MG TABS', 'drug': 'LISINOPRIL', 'date_of_service': '2022-03-25', 'days_sup': 90},
    {'pt_id': 'SAMEDRUG', 'drug_name': 'LISINOPRIL 10 MG TABS', 'drug': 'LISINOPRIL', 'date_of_service': '2022-07-05', 'days_sup': 90},
    {'pt_id': 'SAMEDRUG', 'drug_name': 'LISINOPRIL 10 MG TABS', 'drug': 'LISINOPRIL', 'date_of_service': '2022-09-25', 'days_sup': 90},
    {'pt_id': 'DIFFDRUG', 'drug_name': 'LISINOPRIL 10 MG TABS', 'drug': 'LISINOPRIL', 'date_of_service': '2022-01-01', 'days_sup': 90},
    {'pt_id': 'DIFFDRUG', 'drug_name': 'LOSARTAN 25 MG TABS', 'drug': 'LOSARTAN', 'date_of_service': '2022-03-25', 'days_sup': 90},
    {'pt_id': 'DIFFDRUG', 'drug_name': 'LOSARTAN 25 TABS', 'drug': 'LOSARTAN', 'date_of_service': '2022-07-05', 'days_sup': 90},
    {'pt_id': 'DIFFDRUG', 'drug_name': 'LOSARTAN 25 MG TABS', 'drug': 'LOSARTAN', 'date_of_service': '2022-09-25', 'days_sup': 90},
    {'pt_id': 'NONADH', 'drug_name': 'LISINOPRIL 10 MG TABS', 'drug': 'LISINOPRIL', 'date_of_service': '2022-01-01', 'days_sup': 90},
    {'pt_id': 'NONADH', 'drug_name': 'LISINOPRIL 10 MG TABS', 'drug': 'LISINOPRIL', 'date_of_service': '2022-03-25', 'days_sup': 90},
    {'pt_id': 'NONADH', 'drug_name': 'LISINOPRIL 10 MG TABS', 'drug': 'LISINOPRIL', 'date_of_service': '2022-09-25', 'days_sup': 90},
    {'pt_id': 'COMBPROD', 'drug_name': 'LISINOPRIL 10 MG TABS', 'drug': 'LISINOPRIL', 'date_of_service': '2022-01-01', 'days_sup': 90},
    {'pt_id': 'COMBPROD', 'drug_name': 'LISINOPRIL 10 MG / HYDROCHLOROTHIAZIDE 12.5 MG TABS', 'drug': 'LISINOPRIL', 'date_of_service': '2022-03-25', 'days_sup': 90},
    {'pt_id': 'COMBPROD', 'drug_name': 'LISINOPRIL 10 MG / HYDROCHLOROTHIAZIDE 12.5 MG TABS', 'drug': 'LISINOPRIL', 'date_of_service': '2022-07-05', 'days_sup': 90},
    {'pt_id': 'COMBPROD', 'drug_name': 'LISINOPRIL 10 MG / HYDROCHLOROTHIAZIDE 12.5 MG TABS', 'drug': 'LISINOPRIL', 'date_of_service': '2022-09-25', 'days_sup': 90},
    {'pt_id': 'CONCUSE', 'drug_name': 'LISINOPRIL 10 MG TABS', 'drug': 'LISINOPRIL', 'date_of_service': '2022-01-02', 'days_sup': 90},
    {'pt_id': 'CONCUSE', 'drug_name': 'LISINOPRIL 10 MG TABS', 'drug': 'LISINOPRIL', 'date_of_service': '2022-03-25', 'days_sup': 90},
    {'pt_id': 'CONCUSE', 'drug_name': 'LOSARTAN 25 MG TABS', 'drug': 'LOSARTAN', 'date_of_service': '2022-03-25', 'days_sup': 90},
    {'pt_id': 'CONCUSE', 'drug_name': 'LOSARTAN 25 MG TABS', 'drug': 'LOSARTAN', 'date_of_service': '2022-07-05', 'days_sup': 90},
    {'pt_id': 'CONCUSE', 'drug_name': 'LOSARTAN 25 MG TABS', 'drug': 'LOSARTAN', 'date_of_service': '2022-09-25', 'days_sup': 90}
]
```

We need to convert the dates of service to dates.


```python
for item in rx_clms:
    item['date_of_service'] = dt.datetime.fromisoformat(item['date_of_service'])
```

We then import the sample data into a dataframe and create a column for the DateIndexedArray. The DateIndexedArray is initialized with the date of service and days supply.


```python
import pandas as pd

df = pd.DataFrame(rx_clms)
df['days_covered'] = df.apply(lambda x: DateIndexedArray(x['date_of_service'], x['days_sup']), 1)
```

Next, we calculate the days covered. First, we group by the patient and drug and identify the days covered after adjusting for the overlapping days supply. Then, we regroup by patient and identify the days covered across all drugs (without adjusting for overlap).


```python
df = df \
    .groupby(['pt_id','drug'])['days_covered'] \
    .apply(get_days_covered, True) \
    .reset_index() \
    .groupby('pt_id')['days_covered'] \
    .apply(get_days_covered, False) \
    .reset_index()
```

Now, we can calculate identify the days covered within the treatment period (based upon the start and end of the measurement year).


```python
df['tx_period'] = df['days_covered'] \
    .apply(get_tx_period, args=(dt.datetime(2022, 1, 1, 0, 0),dt.datetime(2022, 12, 31, 0, 0)))
```

From there, we calculate the count of days in the treatment period.


```python
df['tot_days_in_tx_period'] = df['tx_period'] \
    .apply(get_tot_days_in_tx_period)
```

We can also calculate the count of days covered in the treatment period.


```python
df['tot_days_covered'] = df['tx_period'] \
    .apply(get_tot_days_covered, args=(1,))
```

Lastly, we can calculate the PDC by dividing the days covered by the days in the treatment period.


```python
df['pdc'] = df['tot_days_covered']/df['tot_days_in_tx_period']
```

We did it! Take a peak at the fruits of our labor.


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pt_id</th>
      <th>days_covered</th>
      <th>tx_period</th>
      <th>tot_days_in_tx_period</th>
      <th>tot_days_covered</th>
      <th>pdc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>COMBPROD</td>
      <td>01/01/2022: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. ...</td>
      <td>01/01/2022: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. ...</td>
      <td>365</td>
      <td>360</td>
      <td>0.986301</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CONCUSE</td>
      <td>01/02/2022: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. ...</td>
      <td>01/02/2022: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. ...</td>
      <td>364</td>
      <td>360</td>
      <td>0.989011</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DIFFDRUG</td>
      <td>01/01/2022: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. ...</td>
      <td>01/01/2022: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. ...</td>
      <td>365</td>
      <td>353</td>
      <td>0.967123</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NONADH</td>
      <td>01/01/2022: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. ...</td>
      <td>01/01/2022: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. ...</td>
      <td>365</td>
      <td>270</td>
      <td>0.739726</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SAMEDRUG</td>
      <td>01/01/2022: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. ...</td>
      <td>01/01/2022: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. ...</td>
      <td>365</td>
      <td>360</td>
      <td>0.986301</td>
    </tr>
  </tbody>
</table>
</div>



<script>
window.addEventListener('load', function() {
	let message = { height: document.body.scrollHeight, width: document.body.scrollWidth };	

	// window.top refers to parent window
	window.top.postMessage(message, "*");
});
</script>
