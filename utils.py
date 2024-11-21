import pickle
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from itertools import chain
from scipy.interpolate import interp1d
from sklearn.linear_model import QuantileRegressor
from sklearn.utils.fixes import parse_version, sp_version
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import scienceplots



def loadFeaturesandLabels(pathtype,source):

    WindDataset=pd.read_csv(f"data/dataset/{pathtype}/{source}/WindDataset.csv")
    SolarDataset=pd.read_csv(f"data/dataset/{pathtype}/{source}/SolarDataset.csv")

    features_wind=WindDataset.iloc[:,:-1].values
    labels_wind=WindDataset.iloc[:,-1].values

    features_solar=SolarDataset.iloc[:,:-1].values
    labels_solar=SolarDataset.iloc[:,-1].values

    return features_wind,labels_wind,features_solar,labels_solar


#============================Evaluation=================================
def pinball(y,y_hat,alpha):

    return ((y-y_hat)*alpha*(y>=y_hat) + (y_hat-y)*(1-alpha)*(y<y_hat)).mean()

def getRevenue(Xb,Xa,yd,ys):
    return yd*Xb+(Xa-Xb)*ys-0.07*(Xa-Xb)**2


def meanPinballLoss(y_true, y_pred):
    mpd=0
    for i in range(10,100,10):
        mpd+=pinball(y_true,y_pred[f"q{i}"],alpha=i/100)
    return mpd/9

#============================Solar Post-Processing=======================
def forecastWithoutPostProcessing(modelling_table):

    #Extract features
    features_solar=modelling_table.iloc[:,:-1]

    #Forecasting without post-processing
    Models_solar={}
    for quantile in chain([0.1],range(1,100,1),[99.9]):
        with open(f"models/LGBM/full/dwd/solar_q{quantile}.pkl","rb") as f:
            Models_solar[f"q{quantile}"]=pickle.load(f)
    predictions = {}
    for quantile in chain([0.1], range(1, 100, 1), [99.9]):
        output_solar = Models_solar[f"q{quantile}"].predict(features_solar)
        output_solar[output_solar < 1e-2] = 0
        output_solar[modelling_table["hours"] <= 3] = 0
        output_solar[modelling_table["hours"] >= 21] = 0
        predictions[f"q{quantile}"] = output_solar
    predictions_df = pd.DataFrame(predictions)
    modelling_table = pd.concat([modelling_table, predictions_df], axis=1)

    #Expand features
    predictions={}
    for quantile in chain([0.1],range(1,100,1),[99.9]):
        predictions[f"q{quantile}^2"]=modelling_table[f"q{quantile}"]**2
    predictions_df=pd.DataFrame(predictions)
    modelling_table=pd.concat([modelling_table,predictions_df],axis=1)

    return modelling_table

def forecastWithoutPostProcessing_bidding(modelling_table):
    
    features_solar=modelling_table.iloc[:,:-1]
    model_bidding=pickle.load(open("models/LGBM/full/dwd/solar_bidding.pkl","rb"))
    output_solar=model_bidding.predict(features_solar)
    output_solar[output_solar<1e-2]=0
    output_solar[modelling_table["hours"]<=3]=0
    output_solar[modelling_table["hours"]>=21]=0
    modelling_table["Predicted"]=output_solar
    modelling_table["Predicted^2"]=modelling_table["Predicted"]**2

    return modelling_table

def trainPostProcessModel(modelling_table):

    modelling_table=forecastWithoutPostProcessing(modelling_table)

    #Train online post-processing model
    solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"
    Models_revised={}
    for quantile in chain([0.1],tqdm(range(1,100,1)),[99.9]):
        
        params={
            "quantile":quantile/100,
            "alpha":0.1,
            "solver":solver
        }

        model_revised=QuantileRegressor(**params)

        features=modelling_table[[f"q{quantile}",f"q{quantile}^2"]].values
        labels=modelling_table["Solar_MWh_credit"].values
        model_revised.fit(features,labels)

        Models_revised[f"q{quantile}"]=model_revised

    return Models_revised

def trainPostProcessModel_bidding(modelling_table):

    modelling_table=forecastWithoutPostProcessing_bidding(modelling_table)

    params={
        "alpha":0.5
    }
    model_revised=Lasso(**params)
    features=modelling_table[["Predicted","Predicted^2"]].values
    labels=modelling_table["Solar_MWh_credit"].values
    model_revised.fit(features,labels)

    return model_revised

def generateRollingPostProcessCoffs(track="Forecasting"):

    modelling_table=pd.read_csv("data/dataset/latest/dwd/SolarDataset.csv")

    for i in range(int(modelling_table.shape[0]/48-10)):
        modelling_table_online=modelling_table[0:480+i*48].copy()
        if track=="Forecasting":
            Models_revised=trainPostProcessModel(modelling_table_online)
        elif track=="Trading":
            Models_revised=trainPostProcessModel_bidding(modelling_table_online)
        else:
            raise ValueError("Invalid track")
        if not os.path.exists(f"models/Rolling_PostProcess/{track}"):
            os.makedirs(f"models/Rolling_PostProcess/{track}")
        with open(f"models/Rolling_PostProcess/{track}/solar_{i}.pkl","wb") as f:
            pickle.dump(Models_revised,f)


def plotPowerGeneration(Generation_forecast,labels,filename, x_range0,step_size=600,ptype="solar"):
    
    x_range1=x_range0+step_size
    x_range=np.arange(x_range0,x_range1)
    plt.style.use('science')
    plt.figure(figsize=(12,7))
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.rcParams.update({'font.size': 25})
    l1=plt.plot(x_range,labels[x_range],label="true",linewidth=2,color="b")
    colors=["#cf6a87","#f8a5c2"]
    l2=plt.fill_between(x_range,Generation_forecast["q10"][x_range],Generation_forecast["q30"][x_range],color=colors[0],alpha=0.5)
    l3=plt.fill_between(x_range,Generation_forecast["q30"][x_range],Generation_forecast["q50"][x_range],color=colors[1],alpha=0.5)
    plt.fill_between(x_range,Generation_forecast["q50"][x_range],Generation_forecast["q70"][x_range],color=colors[1],alpha=0.5)
    plt.fill_between(x_range,Generation_forecast["q70"][x_range],Generation_forecast["q90"][x_range],color=colors[0],alpha=0.5)

    plt.legend([l1[0],l2,l3],["True","q10-q30, q70-q90","q30-q70"],frameon=False,bbox_to_anchor=(1, 1.15),ncol=3)

    plt.xticks(np.arange(x_range0,x_range1+60,step=60),labels=np.arange(0,(x_range1-x_range0)/2+30,step=30).astype(int))
    
    if ptype=="total":
        plt.yticks(np.arange(0,1400,step=200))
    
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.tick_params(width=2)

    plt.xlabel("Time(h)")
    if ptype=="solar":
        plt.ylabel("Solar Power(MWh)")
    elif ptype=="total":
        plt.ylabel("Total Power(MWh)")

    plt.savefig(f"figs/{filename}",dpi=660)

#==========================Forecasting=================================
def quantile_sort(forecast):

    quantiles = forecast.keys() #sorted(forecast.keys(), key=lambda x: int(x[1:]))
    forecast_array = np.array([forecast[q] for q in quantiles])
    forecast_array_sorted = np.sort(forecast_array, axis=0)
    for i, q in enumerate(quantiles):
        forecast[q] = forecast_array_sorted[i, :]
    
    return forecast

def forecast_wind(wind_features,full,WLimit,availableCapacities=None,model_name="LGBM",source="dwd"):

    if full:
        path="full"
    else:
        path="train"
 
    Wind_Generation_Forecast={}
    coff_ac=np.linspace(0.92,1,101)
    for idx,quantile in enumerate(chain([0.1],range(1,100,1),[99.9])):

        with open(f"models/{model_name}/{path}/{source}/wind_q{quantile}.pkl","rb") as f:
            model=pickle.load(f)

        output=model.predict(wind_features)
        output[output<0]=0

        #Plan Outage
        if WLimit:
            output=np.where(output>coff_ac[idx]*availableCapacities,coff_ac[idx]*availableCapacities,output)

        Wind_Generation_Forecast[f"q{quantile}"]=output
        
    #Quantile Sort
    Wind_Generation_Forecast=quantile_sort(Wind_Generation_Forecast)

    return Wind_Generation_Forecast

def forecast_wind_ensemble(wind_features_dwd,wind_features_gfs,full,WLimit,model_name="LGBM",availableCapacities=None):
    
    if full:
        path="full"
    else:
        path="train"
    
    #Separate Forecasting for dwd and gfs
    temp_dict={}
    for source in ["dwd","gfs"]:
        configs={
            "wind_features":wind_features_dwd if source=="dwd" else wind_features_gfs,
            "full":full,
            "source":source,
            "WLimit":False  # Ensemble first and then truncated
        }
        Wind_Generation_Forecast=forecast_wind(**configs)
        
        for quantile in chain([0.1],range(1,100,1),[99.9]):
            temp_dict[f"{source}_wind_q{quantile}"]=Wind_Generation_Forecast[f"q{quantile}"]
    modelling_table=pd.DataFrame(temp_dict)
    
    #Ensemble Forecasting
    Wind_Generation_Forecast={}
    coff_ac=np.linspace(0.92,1,101)
    for idx,quantile in enumerate(chain([0.1],range(1,100,1),[99.9])):
        with open(f"models/Ensemble/{path}/wind_q{quantile}.pkl","rb") as f:
            model=pickle.load(f)
        feature_meta=modelling_table[[f"dwd_wind_q{quantile}",f"gfs_wind_q{quantile}"]].values
        #Forecast
        output=model.predict(feature_meta)
        output[output<0]=0
        
        #Plan Outage    
        if WLimit:
            output=np.where(output>coff_ac[idx]*availableCapacities,coff_ac[idx]*availableCapacities,output)
        
        Wind_Generation_Forecast[f"q{quantile}"]=output
    
    #Quantile Sort
    Wind_Generation_Forecast=quantile_sort(Wind_Generation_Forecast)

    return Wind_Generation_Forecast


def forecast_solar(solar_features,hours,full,SolarRevise,rolling_test=False,source="dwd",model_name="LGBM"):
    
    if full:
        path="full"
    else:
        path="train"

    Solar_Generation_Forecast={}
    for idx,quantile in enumerate(chain([0.1],range(1,100,1),[99.9])):

        #Load models
        with open(f"models/{model_name}/{path}/{source}/solar_q{quantile}.pkl","rb") as f:
            model=pickle.load(f)
        
        #Forecasting
        output=model.predict(solar_features)
        output[output<1e-2]=0

        #Power at night is set to 0
        output[hours<=3]=0
        output[hours>=21]=0
        
        #Online Post-Processing
        if SolarRevise:
            if rolling_test:

                if not os.path.exists("models/Rolling_PostProcess/Forecasting"):
                    os.makedirs("models/Rolling_PostProcess/Forecasting")

                if not os.listdir("models/Rolling_PostProcess/Forecasting"):
                    generateRollingPostProcessCoffs()

                for i in range(int(output.shape[0]/48-10)): # No corrections 10 days prior
                    with open(f"models/Rolling_PostProcess/Forecasting/solar_{i}.pkl","rb") as f:
                        Model_revised=pickle.load(f)
                    output_solar_origin=output[480+i*48:480+(i+1)*48].reshape(-1,1).copy()
                    features_revised=np.concatenate([output_solar_origin,output_solar_origin**2],axis=1)
                    output[480+i*48:480+(i+1)*48]=Model_revised[f"q{quantile}"].predict(features_revised)
            
            # For Competition use
            else:
                with open(f"models/Revised/solar_q{quantile}.pkl","rb") as f:
                    model_solar_revised=pickle.load(f)
                output_solar_origin=output.reshape(-1,1).copy()
                features_revised=np.concatenate([output_solar_origin,output_solar_origin**2],axis=1)
                output=model_solar_revised.predict(features_revised)
            
            output[output<1e-2]=0

            #Power at night is set to 0
            output[hours<=3]=0
            output[hours>=21]=0
        
        Solar_Generation_Forecast[f"q{quantile}"]=output
    
    #Quantile Sort
    Solar_Generation_Forecast=quantile_sort(Solar_Generation_Forecast)

    return Solar_Generation_Forecast

def forecast_solar_ensemble(solar_features_dwd,solar_features_gfs,hours,full,model_name,SolarRevise):
    
    if full:
        path="full"
    else:
        path="train"
    
    #Separate Forecasting for dwd and gfs
    modelling_table=pd.DataFrame()
    for source in ["dwd","gfs"]:
        configs={
            "solar_features":solar_features_dwd if source=="dwd" else solar_features_gfs,
            "hours":hours,
            "full":full,
            "model_name":model_name,
            "source":source,
            "SolarRevise":SolarRevise
        }
        Solar_Generation_Forecast=forecast_solar(**configs)
        for quantile in range(10,100,10):
            modelling_table[f"{source}_solar_q{quantile}"]=Solar_Generation_Forecast[f"q{quantile}"]
    
    #Ensemble Forecasting
    Solar_Generation_Forecast={}
    for quantile in range(10,100,10):
        with open(f"models/Ensemble/{path}/solar_q{quantile}.pkl","rb") as f:
            model=pickle.load(f)
        feature_meta=modelling_table[[f"dwd_solar_q{quantile}",f"gfs_solar_q{quantile}"]]
        #Forecast
        output=model.predict(feature_meta)
        output[output<1e-2]=0

        Solar_Generation_Forecast[f"q{quantile}"]=output
    
    #Quantile Sort
    Solar_Generation_Forecast=quantile_sort(Solar_Generation_Forecast)

    return Solar_Generation_Forecast

def forecast_total(wind_features_dwd,wind_features_gfs,solar_features,hours,full,WLimit,SolarRevise,availableCapacities=None,rolling_test=False,dx=1e-1,aggregation="True"):
    
    params_wind={
        "wind_features_dwd":wind_features_dwd,
        "wind_features_gfs":wind_features_gfs,
        "full":full,
        "WLimit":WLimit,
        "availableCapacities":availableCapacities
    }

    params_solar={
        "solar_features":solar_features,
        "hours":hours,
        "full":full,
        "SolarRevise":SolarRevise,
        "rolling_test":rolling_test
    }

    Wind_Generation_Forecast=forecast_wind_ensemble(**params_wind)
    Solar_Generation_Forecast=forecast_solar(**params_solar)

    if aggregation:
        Total_generation_forecast=QuantileAdd(Wind_Generation_Forecast,Solar_Generation_Forecast,dx)
    else:
        Total_generation_forecast={}
        for quantile in range(10,100,10):
            Total_generation_forecast[f"q{quantile}"]=Wind_Generation_Forecast[f"q{quantile}"]+Solar_Generation_Forecast[f"q{quantile}"]

    return Total_generation_forecast,Wind_Generation_Forecast,Solar_Generation_Forecast

def QuantileAdd(quantile_values_X,quantile_values_Y,dx):
    quantiles = quantile_values_X.keys()
    forecast_array_X = np.array([quantile_values_X[q] for q in quantiles])
    forecast_array_Y = np.array([quantile_values_Y[q] for q in quantiles])
    Total_generation_forecast={f"q{quantile}": np.zeros(forecast_array_X.shape[1]) for quantile in range(10,100,10)}
    quantiles = np.arange(0,101,1)
    for i in range(forecast_array_X.shape[1]):
        
        X_values=forecast_array_X[:,i]
        Y_values=forecast_array_Y[:,i]
        
        #If the PV intervals are small then just add them up
        if Y_values[90]>600:
            F_x,f_X,x_values=getPDF(quantiles,X_values,dx)
            F_y,f_Y,y_values=getPDF(quantiles,Y_values,dx)
             
            f_Z = np.convolve(f_X, f_Y, mode='full') * dx

            # Determine the new domain for Z
            z_min = x_values[0] + y_values[0]
            z_max = x_values[-1] + y_values[-1]
            z_range_0 = np.arange(z_min, z_max, dx)
            z_range_1 = np.arange(z_min, z_max+dx, dx)
            
            # Make sure f_Z is the same length as z_range_1 and z_range_0.
            if len(f_Z)!=len(z_range_0) and len(f_Z)!=len(z_range_1):
                print("Error")
                
            try:
                f_Z /= np.trapz(f_Z, z_range_0)
                z_range=z_range_0
            except:
                f_Z /= np.trapz(f_Z, z_range_1)
                z_range=z_range_1

            F_Z = np.cumsum(f_Z) * dx
            
            #Extract target quantiles value
            for quantile in range(10,100,10):
                Total_generation_forecast[f"q{quantile}"][i]=np.interp(quantile/100, F_Z, z_range)
        
        else:
            
            for quantile in range(10,100,10):
                Total_generation_forecast[f"q{quantile}"][i]=quantile_values_X[f"q{quantile}"][i]+quantile_values_Y[f"q{quantile}"][i]
    
    return Total_generation_forecast

def getPDF(quantiles,quantile_values,dx):

    probabilities = quantiles / 100.0
    cdf = interp1d(quantile_values, probabilities, kind='linear') 
    x_values = np.arange(quantile_values.min(), quantile_values.max(), dx)
    cdf_values = cdf(x_values)
    pdf_values = np.gradient(cdf_values, x_values)
    pdf_values = pdf_values / np.trapz(pdf_values, x_values)
    
    return cdf_values,pdf_values,x_values

#=============================Bidding=================================

def forecast_bidding_wind(wind_features,full,source,WLimit,model_name="LGBM",availableCapacity=None):

    #Load models
    if full:
        path="full"
    else:
        path="train"
    with open(f"models/{model_name}/{path}/{source}/wind_bidding.pkl","rb") as f:
        model=pickle.load(f)

    #Forecast
    output=model.predict(wind_features)
    output[output<0]=0
    
    #Plan Outage
    if WLimit:
        output=np.where(output>0.95*availableCapacity,0.95*availableCapacity,output)
        
    return output

def forecast_bidding_wind_ensemble(wind_features_dwd,wind_features_gfs,full,WLimit,model_name="LGBM",availableCapacity=None):
        
    if full:
        path="full"
    else:
        path="train"
    
    temp_dict={}
    for source in ["dwd","gfs"]:
        configs={
            "wind_features":wind_features_dwd if source=="dwd" else wind_features_gfs,
            "full":full,
            "source":source,
            "WLimit":WLimit,
            "availableCapacity":availableCapacity
        }
        temp_dict[f"wind_bidding_{source}"]=forecast_bidding_wind(**configs)
    modelling_table=pd.DataFrame(temp_dict)
        
    '''
    with open(f"models/Ensemble/{path}/wind_bidding.pkl","rb") as f:
        model=pickle.load(f)
    '''
    feature_meta=modelling_table[["wind_bidding_dwd","wind_bidding_gfs"]]

    output=feature_meta.iloc[:,0]*0.58+feature_meta.iloc[:,1]*0.42
    output[output<0]=0
    
    return output

def forecast_bidding_solar(solar_features,hours,full,SolarRevise,rolling_test=False,source="dwd",model_name="LGBM"):
    
    if full:
        path="full"
    else:
        path="train"

    with open(f"models/{model_name}/{path}/{source}/solar_bidding.pkl","rb") as f:
        model=pickle.load(f)
    output=model.predict(solar_features)
    output[output<1e-2]=0
    output[hours<=3]=0
    output[hours>=21]=0

    if SolarRevise:
        if rolling_test:
            
            if not os.path.exists("models/Rolling_PostProcess/Trading"):
                os.makedirs("models/Rolling_PostProcess/Trading")

            if not os.listdir("models/Rolling_PostProcess/Trading"):
                generateRollingPostProcessCoffs(track="Trading")

            for i in range(int(output.shape[0]/48-10)):
                with open(f"models/Rolling_PostProcess/Trading/solar_{i}.pkl","rb") as f:
                    Model_revised=pickle.load(f)
                output_solar_origin=output[480+i*48:480+(i+1)*48].reshape(-1,1).copy()
                features_revised=np.concatenate([output_solar_origin,output_solar_origin**2],axis=1)
                output[480+i*48:480+(i+1)*48]=Model_revised.predict(features_revised)

        # For Competition use
        else:
            with open(f"models/Revised/solar_bidding.pkl","rb") as f:
                model_solar_revised=pickle.load(f)
            output_solar_origin=output.reshape(-1,1).copy()
            features_revised=np.concatenate([output_solar_origin,output_solar_origin**2],axis=1)
            output=model_solar_revised.predict(features_revised)

        output[output<1e-2]=0

        output[hours<=3]=0
        output[hours>=21]=0

    return output


def forecast_bidding(wind_features_dwd,wind_features_gfs,solar_features,full,hours,WLimit,SolarRevise,availableCapacity=None,rolling_test=False):

    params_wind={
        "wind_features_dwd":wind_features_dwd,
        "wind_features_gfs":wind_features_gfs,
        "full":full,
        "WLimit":WLimit,
        "availableCapacity":availableCapacity
    }

    params_solar={
        "solar_features":solar_features,
        "hours":hours,
        "full":full,
        "SolarRevise":SolarRevise,
        "rolling_test":rolling_test
    }

    wind_bidding=forecast_bidding_wind_ensemble(**params_wind)
    solar_bidding=forecast_bidding_solar(**params_solar)
    total_bidding=wind_bidding+solar_bidding

    return total_bidding
