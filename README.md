# Wrangle-and-Analyze-Data
Gathering data of we rate dogs twitter page ,analyzing and sharing some insights about it

In this project I Used Python and its libraries like pandas, numpy, matplotlib to gather data from a variety of sources and in a variety of formats like csv,tsv and json files, assess its quality and tidiness, then clean it and then analyze this dataset in order to understand the patterns in it and share some insights that would make us understand the data we have we in a deep way.

This project could be summed up into 4 phases

Phase 1 : Gathering data 
- I downloaded the csv file manualy and then opened it into a data frame.
- I downladed the second file image-predictions.tsv from udacity's server using the provided link and then opened it into a dataframe.
- The third file is a json file which opened then extracted the columns that i wanted from it into a data frame.

Phase 2 : Assessing data

I assessed the data both manualy and programmatically using python different functions like .head() , .info() , and .describe()
The approach that I used in this phase is detect the issues in the data of both types quality issues and tidiness issues and then documented these issuse to later solve them in the data cleaning phase.

Phase 3 : Cleaning data
This step is mainly based on the previous step because here i will solve the issues that i already figured out in the previous phase, the approach used in this step is as follow : Define a solution for the issue you found , Code then solution you defined and then Test the solution.

at this step i figured out that there were some issues in the data that would hinder my analysis and i forget to document it in phase 2 so i iterated the assessing data phase whenever i saw needed.

Phase 4 : Storing and visualizing data

At the is step I merged the three data frames that i have into a single data frame to perform my analysis on, then i started to make the needed visualizations and functions that would help me leverage the insights , patterns and the detalis in my data



