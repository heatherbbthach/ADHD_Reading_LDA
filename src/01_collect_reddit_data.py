# -*- coding: utf-8 -*-
"""

# Latent Dirichlet Allocation (LDA) Analysis of Reddit Posts and Comments

## Part 1: Pull Data from Reddit API and Save to CSV

### Step 1: Install and Import Required Libraries
"""

# Install dependencies using requirements.txt (see README.md)

import pandas as pd
import praw
import csv

"""### Step 2: Authenticate into Reddit API using PRAW"""

#Define a function to read Reddit API credentials from a CSV file
def read_credentials(csv_file):                       #Argument is a CSV file
    credentials_df = pd.read_csv(csv_file)            #Read in the CSV with Pandas, create a dafaframe, and name it "credentials_df"
    row = credentials_df.iloc[0]                      #Pull the first row of data from the dataframe and name it "row"
    return {                                          #Return a dictionary with the specified keys and corresponding values from the dataframe
            "client_id": row["client_id"],
            "client_secret": row["client_secret"],
            "password": row["password"],
            "user_agent": row["user_agent"],
            "username": row["username"]
        }

#Call the function
csv_file = "credentials.csv"                          #Define the file path for the CSV to be read
credentials_dict = read_credentials(csv_file)         #Run the function and name the output dictionary "credentials_dict"

#Use the Output to Authenticate with PRAW
reddit = praw.Reddit(                                 #Create a Reddit instance using PRAW
    client_id=credentials_dict["client_id"],          #Provide the given credentials pulled from the credentials_dict dictionary
    client_secret=credentials_dict["client_secret"],
    password=credentials_dict["password"],
    user_agent=credentials_dict["user_agent"],
    username=credentials_dict["username"],
)

#Test the connection
print("Authenticated as", reddit.user.me())           #If authenticated correctly, this will print the username of the authenticated user.

"""##### Sources for Above Code
PRAW 7.7.1 documentation. (n.d.). https://praw.readthedocs.io/en/stable/

Quick Start - PRAW 7.7.1 documentation. (n.d.). https://praw.readthedocs.io/en/stable/getting_started/quick_start.html

### Step 3: Get Data from Reddit API and Save to CSV
"""

#Define a function to get posts, their comments, and associated data from a Reddit Search
def get_all_search_results_to_csv(reddit, query, csv_filename, batch_size=500):        #Arguments are Reddit instance (defined above), the search query, a CSV filename, and a batch size for processing (predefined)
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:           #Open/Create a CSV file with the given filename in write mode and call it "file"
        writer = csv.writer(file)                                                      #Create a writer object using the CSV library
        writer.writerow(["ID", "Parent ID", "Post Title", "Text", "Post URL", "Type"]) #Write a row with the given strings for the header row
        for submission in reddit.subreddit("all").search(query, limit=batch_size):     #Search for posts that match the query and are within the batch size
            submission_id = submission.id                                              #For each post (or submission), get the ID
            title = submission.title                                                   #Get the title
            text = submission.selftext if submission.selftext else "No Text"           #Get the self text unless it is empty, then return the string
            url = submission.url                                                       #Get the URL
            writer.writerow([submission_id, None, title, text, url, "Post"])           #Write the data to a row with nothing for the Parent ID and "Post" as type

            submission.comments.replace_more(limit=None)                               #Use PRAW's replace_more() function to get all nested comments for a submission
            for comment in submission.comments.list():                                 #For each comment in the submission
                if isinstance(comment, praw.models.Comment):                           #Ensure it's a valid comment
                    comment_id = comment.id                                            #Get the comment ID
                    parent_id = submission.id                                          #Save the ID of the corresponding post as parent_id
                    comment_text = comment.body if comment.body else "Deleted"         #Get the body of the comment unless it is deleted. Then, return the string.
                    writer.writerow([comment_id, parent_id, title, comment_text, url, "Comment"])   #Write the data to a row with "Comment" as the type

    print("Data written to file.")                                                     #Once completed, print this string.

#Call the function
csv_filename = 'data/reddit_adhd_data.csv'                                                  #Define the filename for the output
query = '"ADHD" "Reading Comprehension"'                                               #Define the search query, in this case "ADHD" "Reading Comprehension"
get_all_search_results_to_csv(reddit, query, csv_filename)                             #Call the function with the arguments


"""##### Sources for Above Code
Comment - PRAW 7.7.1 documentation. (n.d.). https://praw.readthedocs.io/en/stable/code_overview/models/comment.html

MoreComments - PRAW 7.7.1 documentation. (n.d.). https://praw.readthedocs.io/en/stable/code_overview/models/more.html#morecomments

PRAW 7.7.1 documentation. (n.d.). https://praw.readthedocs.io/en/stable/

Submission - PRAW 7.7.1 documentation. (n.d.). https://praw.readthedocs.io/en/stable/code_overview/models/submission.html
"""

# Note: credentials.csv is a local-only file and must not be committed to GitHub.
