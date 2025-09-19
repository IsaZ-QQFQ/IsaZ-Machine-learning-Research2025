import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

surfing_data = pd.read_csv("~/PycharmProjects/pythonProject6/.venv/data/df_surf.csv")
if 'surfer_weight_distribution' in surfing_data.columns:
    surfing_data = surfing_data.drop(columns=['surfer_weight_distribution'])
numeric_columns = surfing_data.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    if surfing_data[col].isnull().sum() > 0:
        surfing_data[col] = surfing_data[col].fillna(surfing_data[col].mean())
categorical_columns = surfing_data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    if surfing_data[col].isnull().sum() > 0:
        mode_value = surfing_data[col].mode()
        if len(mode_value) > 0:
            surfing_data[col] = surfing_data[col].fillna(mode_value[0])
print(surfing_data)

from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for column in surfing_data.columns:
    if surfing_data[column].dtype == 'object':
        IS = LabelEncoder()
        surfing_data[column] = IS.fit_transform(surfing_data[column].astype(str))
        label_encoders[column] = IS

X = surfing_data.drop(['board_adequate'], axis=1)
y = surfing_data['board_adequate']

adequate_mapping = {
    "Very inadequate": 0,
    "Inadequate": 1,
    "More or less": 2,
    "Suitable": 3,
    "Very suitable": 4
}
surfing_data['board_adequate'] = surfing_data['board_adequate'].map(adequate_mapping)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.315, random_state=43)

cat_features = X.select_dtypes(include=['object']).columns
num_features = X.select_dtypes(include=['int64', 'float64']).columns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
random_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state = 43))
])
random_model.fit(X_train, y_train)

from sklearn.linear_model import LinearRegression
linear_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
linear_model.fit(X_train, y_train)

from sklearn.metrics import r2_score
y_random_pred = random_model.predict(X_test)
y_random_pred = np.clip(y_random_pred, 0, 4)
print(f"\ny_random_value: {y_random_pred}")
r2_random = r2_score(y_test, y_random_pred)
print(f"\nr2_random: {r2_random}")
y_linear_pred = linear_model.predict(X_test)
y_linear_pred = np.clip(y_linear_pred, 0, 4)
print(f"\ny_linear value: {y_linear_pred}")
r2_linear = r2_score(y_test, y_linear_pred)
print(f"\nr2_linear: {r2_linear}")

from scipy.optimize import minimize
def simple_inverse_prediction(y_goal, pipeline, x_type, x_template):
    def objective(x_value):
        x_temp = x_template.copy()
        x_temp[x_type] = x_value
        y_pre = pipeline.predict(x_temp)[0]
        return (y_pre - y_goal) ** 2

    bounds = (x_template[x_type].min(), x_template[x_type].max())
    result = minimize(objective, x0=x_template[x_type].mean(), bounds=[bounds])
    return result.x[0]

template_row = X_train.mean().to_frame().T
for cat_col in cat_features:
    template_row[cat_col] = X_train[cat_col].mode()[0]

y_goal = 4
x_type = 'board_type'
predicted_value = simple_inverse_prediction(y_goal, random_model, x_type, template_row)
print(predicted_value)

y_goal = 4
x_type = 'wave_height'
predicted_value = simple_inverse_prediction(y_goal, random_model, x_type, template_row)
print(predicted_value)

y_goal = 4
x_type = 'surfer_height'
predicted_value = simple_inverse_prediction(y_goal, random_model, x_type, template_row)
print(predicted_value)

predicted_board_type_code = round(3.125)
decoded_board_type = label_encoders['board_type'].inverse_transform([predicted_board_type_code])[0]
print(decoded_board_type)