### Regional info (zip codes) on selected federal states
import pandas as pd

# Federal states served by the 50Hertz transmission grid
fed_states = [  "Brandenburg",
                "Berlin",
                "Hamburg",
                "Mecklenburg-Vorpommern",
                "Sachsen",
                "Sachsen-Anhalt",
                "Thüringen"
             ]

# Get corresponding zip codes (data from https://github.com/zauberware/postal-codes-json-xml-csv)
zip_data = pd.read_csv("C:/Users/piasw/Desktop/WBS/final project/data/zipcodes.csv", delimiter=",")
zip_data = zip_data[["zipcode", "state"]]
zip_data.columns = ["Postleitzahl", "Bundesland"]

# Some zip codes span multiple federal states. Chose the one that covers more area.
fix_zips = {
    12529: "Brandenburg",
    69434: "Hessen",
    17337: "Brandenburg",
    19357: "Brandenburg",
    59969: "Nordrhein-Westfalen",
    65391: "Hessen",
    21039: "Hamburg",
    22145: "Schleswig-Holstein",
    14715: "Brandenburg",
    7919:  "Sachsen",
    19273: "Niedersachsen",
}
for key, value in fix_zips.items():
    zip_data.loc[zip_data["Postleitzahl"]==key, "Bundesland"] = value
    
zip_data = zip_data.drop_duplicates()
zip_list = zip_data.loc[zip_data["Bundesland"].isin(fed_states), "Postleitzahl"].to_list()