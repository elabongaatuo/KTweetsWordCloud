# WordCloud Visualization for Presidential Candidates (2022)
## Curious what words were associated with top two presidential candidates in the recently held 2022 August elections? This project goes about extracting data from Twitter using the Twitter API, cleaning the extracted data and visualizing the data in a wordcloud for each candidate.

# Introduction
With the recnntly concluded elections, I  was curious to see what words were associated with each candidate. This would involve extracting keywords associated with each candidate, cleaning the data of emojis, links and mobile numbers and finally visualizing the results in a wordcloud based of the candidate's potrait photo.

# Project Flow
* Request for data from Twitter API
* Store in pandas dataframe
* Inspect,clean and pre-process data
* Make custom wordcloud visualization


# User Instruction
The project was entirely done in python. You fork this repo and use requirements.txt to install the required packages using the following command on the terminal.

``` pip install -r requirements. ```

To access the Twitter APi, you need to create a  [Twitter Developer Account](https://developer.twitter.com/) - Sign up and apply for a developer account to be able to scrape data from twitter. You will get the API keys. Be sure to not push to git your access tokens.

To remove the background/clear canvas of your choice picture, you can use [Canva](https://www.canva.com/).

# Author
Elabonga Atuo 
[Email](elabongaatuo@gmail.com)

# Known Issues
Twitter API only allows for data scraping upto the past 7 days. Anything older would require a paid account. Hence the lower word count.

# Files
1. wordcloud Folder - this folder contains photos of the candidates before and after wordcloud visualization side by side.
2. WordCloud.py - contains the code to scrape data from Twitter and Visualize it.

