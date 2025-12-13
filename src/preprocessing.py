import pandas as pd
import numpy as np
from sklearn.base import (
    BaseEstimator,
    TransformerMixin
)
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import STL
from sklearn.impute import KNNImputer


class FeatureEngineering(BaseEstimator, TransformerMixin):

    def __init__(self, n_clusters=4, stl_period=24):
        self.n_clusters = n_clusters
        self.stl_period = stl_period

        # Models learned during fit()
        self.cluster_model = None
        self.cluster_features = None

        self.trend_model = None
        self.season_model = None
        self.resid_model = None

        # Cloud map
        self.cloud_map = {
            'Unknown': 1, 'Opaque Ice': 2, 'Overlapping': 3,
            'Super-Cooled Water': 4, 'Cirrus': 5, 'Fog': 6,
            'Water': 7, 'Overshooting': 8,
            'Probably Clear': 9, 'Clear': 10
        }


    def _cloud_mapping(self, df):
        df['Cloud Type'] = df['Cloud Type'].map(self.cloud_map).fillna(1)
        return df


    def _time_features(self, df):
        df = df.copy()

        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.sort_values('Timestamp')

        solar_time = df['Timestamp'] - pd.Timedelta(hours=5)
        solar_hour = solar_time.dt.hour

        df['hour_sin'] = np.sin(2 * np.pi * solar_hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * solar_hour / 24)

        df['month_sin'] = np.sin(2 * np.pi * df['Timestamp'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['Timestamp'].dt.month / 12)

        df['doy'] = solar_time.dt.dayofyear

        return df


    def _astronomy_features(self, df):
        df = df.copy()
        doy = df['doy']

        delta = 23.45 * np.sin(np.radians(360 * (284 + doy) / 365))
        df['solar_declination'] = delta

        B = np.radians((doy - 81) * 360 / 365)
        df['equation_of_time'] = (
            9.87 * np.sin(2*B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
        )

        df['sun_earth_distance_factor'] = 1 + 0.033 * np.cos(np.radians(360 * doy / 365))
        df['extraterrestrial_radiation'] = 1367 * df['sun_earth_distance_factor']

        return df


    def _sun_features(self, df):
        df = df.copy()

        sunrise_dt = pd.to_datetime(df['sunrise'])
        sunset_dt = pd.to_datetime(df['sunset'])

        df['sunHour'] = (sunset_dt - sunrise_dt).dt.total_seconds()

        curr = df['Timestamp'].dt.time
        rise = sunrise_dt.dt.time
        set_  = sunset_dt.dt.time

        df['is_daytime'] = [
            1 if (r <= c <= s) else 0
            for c, r, s in zip(curr, rise, set_)
        ]

        return df


    def _physics_features(self, df):
        df = df.copy()
        eps = 1e-6

        df['clearsky_index'] = df['GHI'] / (df['Clearsky GHI'] + eps)
        df['diffuse_fraction'] = df['DHI'] / (df['GHI'] + eps)
        df['wind_cooling_potential'] = df['windspeedKmph'] / (df['tempC'] + 273.15)

        return df


    def _time_dynamic_features(self, df):
        df = df.copy()

        df['target_time_1h'] = df['Timestamp'] - pd.Timedelta(hours=1)

        lookup = df[['Timestamp', 'GHI', 'cloudcover']].copy()
        lookup.columns = ['ts_ref', 'GHI_lag1', 'cloudcover_lag1']

        df = df.merge(lookup, left_on='target_time_1h',
                      right_on='ts_ref', how='left')

        idx = df.set_index('Timestamp')
        df['GHI_rolling_mean_3h'] = (
            idx['GHI'].rolling('3h', min_periods=1).mean().values
        )

        df.drop(columns=['target_time_1h', 'ts_ref'], inplace=True)

        df[['GHI_lag1','cloudcover_lag1','GHI_rolling_mean_3h']] = \
            df[['GHI_lag1','cloudcover_lag1','GHI_rolling_mean_3h']].fillna(0)

        return df


    def _gradient_features(self, df):
        df = df.copy()

        df['GHI_diff_1h'] = df['GHI'] - df['GHI_lag1']
        df['cloudcover_diff_1h'] = df['cloudcover'] - df['cloudcover_lag1']
        df['clearsky_index_diff'] = df['clearsky_index'].diff().fillna(0)
        df['GHI_acceleration'] = df['GHI_diff_1h'].diff().fillna(0)

        return df


    def _apply_weather_clusters(self, df):
        df = df.copy()
        df['weather_cluster'] = self.cluster_model.predict(
            df[self.cluster_features].fillna(0)
        )
        return df


    def _apply_stl(self, df):
        df = df.copy()
        t = np.arange(len(df)).reshape(-1, 1)

        df['GHI_trend'] = self.trend_model.predict(t)
        df['GHI_seasonal'] = self.season_model.predict(t)
        df['GHI_residual'] = self.resid_model.predict(t)

        return df


    def fit(self, df, y=None):

        df = df.copy()

        # Run basic FE steps
        df = (df.pipe(self._cloud_mapping)
                .pipe(self._time_features)
                .pipe(self._astronomy_features)
                .pipe(self._sun_features)
                .pipe(self._physics_features)
                .pipe(self._time_dynamic_features)
        )

        # Fit clustering model
        self.cluster_features = [
            'cloudcover_lag1',
            'humidity',
            'windspeedKmph',
            'GHI_rolling_mean_3h',
            'clearsky_index',
            'diffuse_fraction'
        ]

        self.cluster_model = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.cluster_model.fit(df[self.cluster_features].fillna(0))

        # Fit STL decomposition on GHI
        stl = STL(df['GHI'].fillna(0), period=self.stl_period).fit()

        t = np.arange(len(df)).reshape(-1, 1)

        self.trend_model  = LinearRegression().fit(t, stl.trend)
        self.season_model = LinearRegression().fit(t, stl.seasonal)
        self.resid_model  = LinearRegression().fit(t, stl.resid)

        return self


    def transform(self, df):

        df = df.copy()

        df = (df.pipe(self._cloud_mapping)
                .pipe(self._time_features)
                .pipe(self._astronomy_features)
                .pipe(self._sun_features)
                .pipe(self._physics_features)
                .pipe(self._time_dynamic_features)
                .pipe(self._gradient_features)
        )

        df = self._apply_weather_clusters(df)
        df = self._apply_stl(df)

        df = df.drop(columns=['Timestamp',
                              'sunrise', 'sunset', 'moonrise', 'moonset',
                              'tempC', 'Pressure', 'DewPointC'], errors='ignore')

        self.feature_names_out_ = df.columns.to_list()

        return df

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


class SolarImputer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.knn = KNNImputer(n_neighbors=3, weights="distance")
        self.solar_cols = ['Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI']

    def _prepare_timestamp(self, df):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        return df.set_index('Timestamp').sort_index()

    def fit(self, df, y=None):
        df = df.copy()

        df_prepared = self._prepare_timestamp(df)

        self.knn.fit(df_prepared[self.solar_cols])
        return self

    def transform(self, df):
        df = df.copy()
        df = self._prepare_timestamp(df)

        # Interpolate solar radiation block
        df[['GHI','DHI','DNI']] = (
            df[['GHI','DHI','DNI']].interpolate(method='time')
        ).ffill().bfill()

        # Apply KNN imputation to clearsky
        df[self.solar_cols] = self.knn.transform(df[self.solar_cols])

        # Cloud type missing Unknown (1)
        df['Cloud Type'] = df['Cloud Type'].fillna('Unknown')

        return df.reset_index()