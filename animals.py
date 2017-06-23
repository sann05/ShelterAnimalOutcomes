import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score

encoder_color = LabelEncoder()
encoder_sex = LabelEncoder()
encoder_animal = LabelEncoder()

#To transform DateTime column
def day_of_week(born):
    return born.weekday()
def get_month(born):
    return born.month
def get_day(born):
    return born.day
#To transoform color
def transform_color(color):
    return color.split('/')[0]
def ismix(breed):
    if 'mix' in breed:
        return 1
    return 0
#To transform AgeuponOutcome column
def calc_age_in_years(x):
    x = str(x)
    if x == 'nan': return 0
    age, period = x.split()
    age = int(age)
    if period.find('year') > -1: return age
    if period.find('month')> -1: return age / 12.
    if period.find('week')> -1: return age / 52.
    if period.find('day')> -1: return age / 365.
    else: return 0

#Getting rid of empties in data
def convert_empties(data):
    data = pd.DataFrame(data)
    data['Name'] = data['Name'].fillna('0')
    data['Color'] = data['Color'].apply(transform_color)

    #Found out that we have one object with empty value at SexuponOutcome. It's not important to leave it
    #in data. It can be error, and event if it's not one sample doesn't give much value.
    data = data[~data['SexuponOutcome'].isnull()]
    return data

def convert_data(data):
    data = pd.DataFrame(data)
    encoder = LabelEncoder()

    #We have probability that existance of name also influes on the data. So transform column Name to
    #0 - no name , 1 - name exists.
    #It turns out that actual name also have pretty big meaning
    data['Name'] = encoder.fit_transform(data['Name'])

    #Cateforize dogs at 10 categories accorting to AKS
    dogs_group = pd.read_csv('data\\breed_info.csv')
    dogs_breeds = set(data[data['AnimalType'] == 'Dog']['Breed'])
    for breed in dogs_breeds:
        if '/' in breed:
            part = breed.split('/')
            group1 = dogs_group[dogs_group['Breed'] == part[0]]['Group'].values[0]
            group2 = dogs_group[dogs_group['Breed'] == part[1]]['Group'].values[0]
            data.loc[data.Breed == breed, group1] = 1
            data.loc[data.Breed == breed, group2] = 1
        else:
            group = dogs_group[dogs_group['Breed'] == breed]['Group'].values[0]
            data.loc[data.Breed == breed, group] = 1
    data['Toy'] = data['Toy'].fillna(0)
    data['Non-Sporting'] = data['Non-Sporting'].fillna(0)
    data['Working'] = data['Working'].fillna(0)
    data['Hound'] = data['Hound'].fillna(0)
    data['Herding'] = data['Herding'].fillna(0)
    data['PittBull'] = data['PittBull'].fillna(0)
    data['Sporting'] = data['Sporting'].fillna(0)
    data['Terrier'] = data['Terrier'].fillna(0)
    data['Unknown'] = data['Unknown'].fillna(0)

    data['Color'] = encoder_color.transform(data['Color'])
    data['SexuponOutcome'] = encoder_sex.transform(data['SexuponOutcome'])
    data['AnimalType'] = encoder_animal.transform(data['AnimalType'])

    #Due debugging we found out that partial time (as Day of week, day of month and month) affect on the final
    #outcome type more.
    data['DayOfWeek'] = pd.to_datetime(data['DateTime']).apply(day_of_week)
    data['Day'] = pd.to_datetime(data['DateTime']).apply(get_day)
    data['Month'] = pd.to_datetime(data['DateTime']).apply(get_month)

    data['AgeuponOutcome'] = data.AgeuponOutcome.apply(calc_age_in_years)

    return data[['Name','DayOfWeek','Day' ,'Month' ,'AnimalType','SexuponOutcome','AgeuponOutcome','Color',
         'Toy','Non-Sporting','Working','Hound','Herding','PittBull','Sporting','Sporting','Terrier','Unknown']]

data = convert_empties(pd.read_csv('data\\train.csv'))
test = convert_empties(pd.read_csv('data\\test.csv'))

#Step 1. Data preparation.
#In the beginning we just transform data

encoder_color.fit(pd.concat([data['Color'], test['Color']]))
encoder_sex.fit(pd.concat([data['SexuponOutcome'], test['SexuponOutcome']]))
encoder_animal.fit(pd.concat([data['AnimalType'], test['AnimalType']]))

encoder = LabelEncoder()
scaler = StandardScaler()

#Categorized text data
X = convert_data(data)
X = scaler.fit_transform(X)
y = encoder.fit_transform(data['OutcomeType'])

#Do scaling
test_pred = convert_data(test)
test_pred = scaler.transform(test_pred)

#Step 2. Realizing algoritm
max_n = 0
max_score = 0
for i in range(5,10):
    model = XGBClassifier(max_depth=i)
    kf = KFold(len(y),n_folds=5,random_state=42, shuffle=True)
    #Using accuracy because of final table using it measure
    score = cross_val_score(model, X, y, cv=kf, scoring='accuracy').mean()
    print('Cross validation score =', score)
    print('max_depth =', i)
    if score > max_score:
        max_score = score
        max_n = i
print('Max Cross validation score =',max_score)
print('Max max_depth =', max_n)
model = XGBClassifier(max_depth=max_n)
model.fit(X,y)
prediction = model.predict_proba(test_pred)

#Just to see what features are important and what are not
print(model.feature_importances_)

#Step 3. Save data to file.
submission = pd.DataFrame({
    "ID": test["ID"],
    "Adoption": prediction[:,0],
    "Died": prediction[:,1],
    "Euthanasia": prediction[:,2],
    "Return_to_owner": prediction[:,3],
    "Transfer": prediction[:,4]

})

submission.to_csv("animals-submission.csv", index=False)