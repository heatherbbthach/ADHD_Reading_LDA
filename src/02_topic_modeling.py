# Part 2: LDA

# Install dependencies using requirements.txt (see README.md)

import pandas as pd
import os
import re
from wordcloud import WordCloud
import gensim
from gensim.utils import simple_preprocess
import nltk
import requests
from gensim.models import CoherenceModel
from gensim.models import LdaModel
import matplotlib.pyplot as plt
import numpy as np
import gensim.corpora as corpora
from pprint import pprint
import pyLDAvis.gensim_models as gensimvis
import pickle
import pyLDAvis

def main():
    # Step 2: Read in the Data
    
    #Define a function to read the data and create a dataframe
    def read_data(csv_file):                                                                        #Pass one argument, csv file
        dataframe = pd.read_csv(csv_file)                                             #Use Pandas to read in the data from CSV and create a dataframe
        dataframe= dataframe.drop(columns=["ID", "Parent ID", "Post URL", "Type"])          #Remove all columns except the body and title                                                             #Print out the first rows of df to check
        return dataframe
    
    #Call the function
    csv_file = 'data/reddit_adhd_data.csv'              #Set the csv_file to pass through as argument
    df = read_data(csv_file)                       #Run the function
    df.head()                                      #Print first few rows
    
    # Source for Above Code
    #Kapadia, S. (2024, April 15). Topic Modeling in Python: Latent Dirichlet Allocation (LDA). Medium. https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
    
    # Step 3: Data Cleaning with Regular Expressions
    
    #Remove web addresses
    df['body_processed'] = df['Text'].map(lambda x: re.sub(r'http\S+|www\S+|https\S+', '', x.lower()) if isinstance(x, str) else '')
    
    #Remove domain-like patterns
    df['body_processed'] = df['body_processed'].map(lambda x: re.sub(r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/\S*)?\b', '', x) if isinstance(x, str) else '')
    
    #Remove line breaks
    df['body_processed'] = df['body_processed'].map(lambda x: re.sub(r'[\r\n]', ' ', x) if isinstance(x, str) else '')
    
    #Remove punctuation
    df['body_processed'] = df['body_processed'].map(lambda x: re.sub(r'[^\w\s]', '', x) if isinstance(x, str) else '')
    
    #Remove all numbers, keeping only letters and spaces
    df['body_processed'] = df['body_processed'].map(lambda x: re.sub(r'[^a-zA-Z\s]', '', x) if isinstance(x, str) else '')
    
    #Convert all text to lowercase
    df['body_processed'] = df['body_processed'].map(lambda x: x.lower() if isinstance(x, str) else '')
    
    #Print out the first rows of df
    df['body_processed'].head()
    
    # Source for Above Code
    # Kapadia, S. (2024, April 15). Topic Modeling in Python: Latent Dirichlet Allocation (LDA). Medium. https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
    
    # Note: The above code uses lambda functions similar to the source code for brevity.
    
    # Step 4: Exploratory Analysis
    
    #Define a function to create the wordcloud
    def create_wordcloud(dataframe, column_name, max_words=1000, bg_color="white", contour_width=3, contour_color='steelblue'):      #Pass the dataframe, column, max_words, background color, contour width and color as arguments
        # Join the text from the specified column into one long string
        long_string = ','.join(list(dataframe[column_name].values))
    
        # Create a WordCloud object
        wordcloud = WordCloud(background_color=bg_color, max_words=max_words, contour_width=contour_width, contour_color=contour_color)
    
        # Generate the word cloud
        wordcloud.generate(long_string)
    
        # Return the word cloud image
        return wordcloud.to_image()
    
    #Call the function
    create_wordcloud(df, "body_processed")
    
    # Source for Above Code
    # Kapadia, S. (2024, April 15). Topic Modeling in Python: Latent Dirichlet Allocation (LDA). Medium. https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
    
    # Step 5: Additional Cleaning & Preparation for Analysis
    
    
    #Load the default stopwords list from the GitHub source
    stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
    stop_words = set(stopwords_list.decode().splitlines())
    
    #Define a set of additional custom stopwords to remove from the text
    additional_words = {"adhd", "read", "reading", "comprehension", "https", "dog", "good", "bad", "gt", "post", "reddit",
                        "great", "glitter", "people", "things", "lot", "thing", "subreddit", "moderators", "posting", "comments",
                        "comment", "amp", "feel", "pretty", "life", "body", "high", "day", "list", "love", "start", "rules", "year", "years",
                        "night", "true", "pvs", "auto", "person", "contact", "pay", "automatically", "bot", "deleted", "glitterex", "message",
                        "money", "book", "books", "food", "dont", "ive", "find", "make", "back", "youre", "long", "didnt","put", "short",
                        "goals", "reported", "hes", "hate", "wanted", "made", "wasnt", "side", "havent", "removed", "resubmit", "radhd",
                        "didn", "buy", "car", "give", "doesnt", "makes", "times", "based", "posts", "stuff", "found", "mind", "theyre",
                        "felt", "basically", "began", "world", "told", "hours", "lol", "live", "ill", "games", "isnt", "bit", "yeah"}
    
    #Update the default stopwords list by adding the above custom set of stopwords
    stop_words.update(additional_words)
    
    #Define a function to convert sentences to words
    def sent_to_words(sentences):                                               #Take one argument, sentences
        for sentence in sentences:                                              #For each sentence
            yield(gensim.utils.simple_preprocess(str(sentence)))                #Tokenize the sentence, i.e. turn it into words
    
    #Define a function to remove stopwords
    def remove_stopwords(texts):                                                #Take one argument, texts
        return [[word for word in simple_preprocess(str(doc))                   #Return each word from the tokenized text if the word is not in the stopwords list
                 if word not in stop_words] for doc in texts]
    
    #Call the function to convert sentences to words
    data = df.body_processed.values.tolist()                                    #Convert the "body_processed" column of text to a list and name it data
    data_words = list(sent_to_words(data))                                      #Execute the function using the data list just created and name the output data_words
    
    #Call the function to remove stopwords
    data_words = remove_stopwords(data_words)
    
    print(data_words[:1][0][:30])                                               #Print the first 30 words from data_words for the first record
    
    ##### Sources for Above Code
    #Kapadia, S. (2024, April 15). Topic Modeling in Python: Latent Dirichlet Allocation (LDA). Medium. https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
    
    #Gupta, Rishabh [rg089]. (2021, July 15). A list of English stopwords. Gist. https://gist.github.com/rg089/35e00abf8941d72d419224cfd5b5925d/
    
    #Step 6: Tuning &  Determining the Optimal Number of Topics
    
    #Create Dictionary for LDA Model
    id2word = corpora.Dictionary(data_words)
    #Create Corpus for LDA Model
    texts = data_words
    #Get Term Document Frequency for LDA Model
    corpus = [id2word.doc2bow(text) for text in texts]
    
    #Define a Function to Calculate Coherence Values
    def compute_coherence_values(dictionary, corpus, texts, coherence, limit, start=1, step=1, passes=50):      #Take dictionary, corpus, texts, coherence method, limit, start, and step as arguments
        coherence_values = []                                                                        #Create a list to store coherence values
        model_list = []                                                                              #Create a list to store models
        topic_coherence_values = []                                                                  #Create a list to store coherence values for each topic per model
        for num_topics in range(start, limit + 1, step):                                             #For each number of topics in the range between the start value and the limit plus 1 (to account for indexing) in increments of 1
            model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes, random_state=42)               #Create an LDA model for the given number of topics
            model_list.append(model)                                                                 #Add the model to the list of models
    
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence=coherence, processes=1)       #Compute coherence for the model
            coherence_values.append(coherencemodel.get_coherence())                                                     #Add the coherence value for the model to the list
            topic_coherence_values.append(coherencemodel.get_coherence_per_topic())                                     #Add the coherence values for the topics to the list
    
        return model_list, coherence_values, topic_coherence_values                                  #Return the list of models, coherence values, and topic coherence values
    
    # Sources for Above Code
    #Kapadia, S. (2024, April 15). Topic Modeling in Python: Latent Dirichlet Allocation (LDA). Medium. https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
    
    #Tran, T. [trnduythanhkhttt]. (2022, August 19). LDA and Topic Coherence Score. Kaggle. https://www.kaggle.com/code/trnduythanhkhttt/lda-and-topic-coherence-score
    
    
    #Call the Function to Calculate Coherence using C_V Method
    limit=15; start=1; step=1;                                                                       #Set the limit, start, and step values
    model_list_cv, coherence_values_cv, topic_coherence_values_cv =  compute_coherence_values(       #Create lists for models, coherence values, and topic coherence values and set equal to the output of the function
        dictionary=id2word,                                                                          #Define the arguments
        corpus=corpus,
        texts=texts,
        coherence="c_v",
        start=start,
        limit=limit,
        step=step,
        passes=50
    )
    
    # Source for Above Code
    # Tran, T. [trnduythanhkhttt]. (2022, August 19). LDA and Topic Coherence Score. Kaggle. https://www.kaggle.com/code/trnduythanhkhttt/lda-and-topic-coherence-score
    
    
    #Call the Function to Calculate Coherence using U_Mass Method
    limit=15; start=1; step=1;                                                                             #Set the limit, start, and step values
    model_list_umass, coherence_values_umass, topic_coherence_values_umass = compute_coherence_values(     #Create lists for models, coherence values, and topic coherence values and set equal to the output of the function
        dictionary=id2word,                                                                                #Define the arguments
        corpus=corpus,
        texts=texts,
        coherence="u_mass",
        start=start,
        limit=limit,
        step=step
    )
    
    # Source for Above Code
    # Tran, T. [trnduythanhkhttt]. (2022, August 19). LDA and Topic Coherence Score. Kaggle. https://www.kaggle.com/code/trnduythanhkhttt/lda-and-topic-coherence-score
    
    #Call the Function to Calculate Coherence using CUCI Method
    limit=15; start=1; step=1;                                                                          #Set the limit, start, and step values
    model_list_cuci, coherence_values_cuci, topic_coherence_values_cuci = compute_coherence_values(     #Create lists for models, coherence values, and topic coherence values and set equal to the output of the function
        dictionary=id2word,                                                                             #Define the arguments
        corpus=corpus,
        texts=texts,
        coherence="c_uci",
        start=start,
        limit=limit,
        step=step
    )
    
    # Source for Above Code
    # Tran, T. [trnduythanhkhttt]. (2022, August 19). LDA and Topic Coherence Score. Kaggle. https://www.kaggle.com/code/trnduythanhkhttt/lda-and-topic-coherence-score
    
    #Call the Function to Calculate Coherence using CNPMI Method
    limit=15; start=1; step=1;                                                                              #Set the limit, start, and step values
    model_list_cnpmi, coherence_values_cnpmi, topic_coherence_values_cnpmi = compute_coherence_values(      #Create lists for models, coherence values, and topic coherence values and set equal to the output of the function
        dictionary=id2word,                                                                                 #Define the arguments
        corpus=corpus,
        texts=texts,
        coherence="c_npmi",
        start=start,
        limit=limit,
        step=step
    )
    
    # Source for Above Code
    # Tran, T. [trnduythanhkhttt]. (2022, August 19). LDA and Topic Coherence Score. Kaggle. https://www.kaggle.com/code/trnduythanhkhttt/lda-and-topic-coherence-score
    
    #Create a function to visualize the results of the Coherence Calculations
    def plot_coherence():
        models=[]                                                     #Create an empty list to create model names
        for i in range(1,limit+1):                                    #For each integer in the range between 1 and the limit plus one
            models.append("model{0}".format(i))                       #Name the model in the format "model{0}" and add it to the list
    
        plt.figure(figsize=(8,4))                                    #Define the figure size
        plt.plot(models,coherence_values_cv,label="c_v")              #Plot the Coherence C_V values
        plt.title("Coherence score")             #Set the plot title
        plt.xlabel("Trained Topic Models")                            #Set the label for the x-axis
        plt.ylabel("Coherence score")                                 #Set the label for the y-axis
        plt.legend()                                                  #Add a legend
        plt.xticks(rotation=45)                                       #Rotate labels on the x-axis by 45 degrees
        plt.yticks(np.arange(.25, .55, 0.1))                            #Set the Y-axis intervals to be between -7 and 1, with 0.25 intervals
        plt.show()                                                    #Show the plot
    
    #Call the function
    plot_coherence()
    
    
    # Create a list of model names
    models = ["model{0}".format(i) for i in range(1, limit + 1)]
    
    # Create a DataFrame to store the coherence values
    df_coherence = pd.DataFrame({
        "Trained Topic Model": models,
        "Coherence Value (Cv)": coherence_values_cv
    })
    
    # Display the table
    print(df_coherence)
    
    # Optionally, save to a CSV file for further analysis
    df_coherence.to_csv("coherence_values_table.csv", index=False)
    
    #Sources for Above Code
    #GeeksforGeeks. (2020, April 12). Matplotlib.pyplot.yticks() in Python. GeeksforGeeks. https://www.geeksforgeeks.org/matplotlib-pyplot-yticks-in-python/
    
    #Tran, T. [trnduythanhkhttt]. (2022, August 19). LDA and Topic Coherence Score. Kaggle. https://www.kaggle.com/code/trnduythanhkhttt/lda-and-topic-coherence-score
    
    # Step 7: Train the Model and Get the Topics
    
    #Define a function to train the LDA Model
    def train_lda_model(corpus, id2word, num_topics, passes):                                                      #Take corpus, id2word, num_topics, and passes as arguments
        return gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics, passes=passes, random_state=42)    #Return the model
    
    #Call the function
    num_topics = 3                                                                                                 #Set the number of topics based on coherence
    passes = 50                                                                                                    #Set the number of passes
    lda_model = train_lda_model(corpus, id2word, num_topics, passes)                                               #Run the function and set output equal to lda_model
    
    # Source for Above Code
    #Kapadia, S. (2024, April 15). Topic Modeling in Python: Latent Dirichlet Allocation (LDA). Medium. https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
    
    
    #Create a function to print the Keyword in the topics
    def print_topics(lda_model, num_words=10):                       #Set lda_model and number of words as arguments
        topics = lda_model.print_topics(num_words=num_words)         #Get the topics from the LDA model and assign them to the topics list with their top 10 words
        for topic in topics:                                         #For each topic
            print(topic)                                             #Print the topic
    
    #Call the function
    print_topics(lda_model)
    
    #Source for Above Code
    #Kapadia, S. (2024, April 15). Topic Modeling in Python: Latent Dirichlet Allocation (LDA). Medium. https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
    
    #Define a function to calculate the coherence of the final model
    def calculate_coherence_final_model(lda_model, texts, dictionary, coherence):                                    #Take the model, texts, dictionary, and coherence method as arguments
        coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence=coherence, processes=1)      #Create the coherence model
        coherence_lda = coherence_model.get_coherence()                                                              #Get the coherence value for the model
        print('\nCoherence Score: ', coherence, coherence_lda)                                                       #Print the coherence score, the coherence method, and value
    
    #Call the Function to Calculation Coherence Score using C_V
    coherence='c_v'                                                                #Define the coherence calculation method
    dictionary=id2word                                                             #Define the dictionary
    calculate_coherence_final_model(lda_model, texts, dictionary, coherence)       #Run the function
    
    #Call the Function to Calculation Coherence Score using U_Mass
    coherence='u_mass'                                                             #Define the coherence calculation method
    dictionary=id2word                                                             #Define the dictionary
    calculate_coherence_final_model(lda_model, texts, dictionary, coherence)       #Run the function
        
    # Function to calculate coherence values for selected topic numbers
    def calculate_selected_coherence(dictionary, corpus, texts, final_model, final_model_topics, coherence="c_v"):
        selected_topics = [1, 2] + list(range(4, 16))  # Include 1, 2, and 4-15 topics
        models = []
        coherence_values = []
    
        for num_topics in selected_topics:
            model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes, random_state=42)
            models.append("model" + str(num_topics))
    
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence=coherence, processes=1)
            coherence_values.append(coherencemodel.get_coherence())
    
        # Get coherence for final model
        final_coherencemodel = CoherenceModel(model=final_model, texts=texts, dictionary=dictionary, coherence=coherence, processes=1)
        final_coherence_value = final_coherencemodel.get_coherence()
    
        # Add final model results to the list
        models.append("Final Model (" + str(final_model_topics) + " topics)")
        coherence_values.append(final_coherence_value)
    
        # Create a DataFrame
        df_coherence = pd.DataFrame({
            "Trained Topic Model": models,
            "Coherence Value (Cv)": coherence_values
        })
    
        # Print the coherence score for the final model
        print("\nFinal Model (" + str(final_model_topics) + " topics) Coherence Score (Cv):", final_coherence_value)
    
        # Display the table
        print("\nCoherence Table:")
        print(df_coherence.to_string(index=False))  # Print table nicely formatted
    
        # Optionally, save to CSV
        df_coherence.to_csv("coherence_values.csv", index=False)
    
        return df_coherence
    
    #Call the function to calculate coherence values for the selected models
    df_coherence_table = calculate_selected_coherence(
        dictionary=id2word,
        corpus=corpus,
        texts=texts,
        final_model=lda_model,
        final_model_topics=lda_model.num_topics,
        coherence="c_v"
    )
    
    
    # Use the C_V coherence values calculated during model evaluation
    models = ["model" + str(i) for i in range(start, limit + 1, step)]
    coherence_values = coherence_values_cv
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Set background to white
    fig.patch.set_facecolor('white')  # White figure background
    ax.set_facecolor('white')  # White background behind axes
    
    
    # Plot the coherence values
    ax.plot(models, coherence_values, marker='o', linestyle='-', label="Cv Coherence", color='black')
    
    # Highlight the final model
    final_model_index = models.index("model3")
    ax.scatter(final_model_index, coherence_values[final_model_index], marker='*', s=150, color='black', label="Cv of Final Model (3 topics)", zorder=3)
    
    # Formatting
    ax.set_title("Coherence Scores for Different Topic Models")
    ax.set_xlabel("Trained Topic Models")
    ax.set_ylabel("Coherence Value (Cv)")
    ax.set_xticks(models)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_yticks(np.arange(min(coherence_values) - 0.02, max(coherence_values) + 0.02, 0.02))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    
    # Show the plot
    plt.show()
    
    #Source for Above Code
    #Tran, T. [trnduythanhkhttt]. (2022, August 19). LDA and Topic Coherence Score. Kaggle. https://www.kaggle.com/code/trnduythanhkhttt/lda-and-topic-coherence-score
    
    #Step 8: Visualize the Results
    
    #Create a function to generate and save a PyLDAvis visualization
    def save_lda_visualization(lda_model, corpus, id2word, num_topics):
        LDAvis_data_filepath = os.path.join('ldavis_prepared_final2'+str(num_topics))             #Create a file path string to save the PyLDAvis data
        LDAvis_html_filepath = os.path.join('ldavis_prepared_final2'+str(num_topics)+'.html')      #Create a file path string to save the visualization as an HTML file
        LDAvis_prepared = gensimvis.prepare(lda_model, corpus, id2word)                          #Prepare the data needed to generate the visualization
        with open(LDAvis_data_filepath, 'wb') as f:                                              #With the filepath open as f
            pickle.dump(LDAvis_prepared, f)                                                      #Write the data for the model to the file
        pyLDAvis.save_html(LDAvis_prepared, LDAvis_html_filepath)                                #Save the PyLDAvis visualization as an HTML file
        print("PyLDAvis visualization created and saved.")                                       #Print this message when complete
    
    #Call the function
    save_lda_visualization(lda_model, corpus, id2word, num_topics)
    
    # Get topic distributions for each document
    doc_topic_distributions = []
    
    for doc in corpus:
        topic_probs = lda_model.get_document_topics(doc, minimum_probability=0.0)  # Ensure all topic probabilities are included
        doc_topic_distributions.append(topic_probs)
    
    
    # Create a list to store topic information
    doc_topic_data = []
    
    for i, topic_probs in enumerate(doc_topic_distributions):
        sorted_topics = sorted(topic_probs, key=lambda x: x[1], reverse=True)  # Sort topics by probability
        dominant_topic, top_prob = sorted_topics[0]  # Get the topic with the highest probability
        doc_topic_data.append({"Document_Index": i, "Dominant_Topic": dominant_topic, "Top_Probability": top_prob})
    
    # Convert list into a DataFrame
    doc_topic_df = pd.DataFrame(doc_topic_data)
    
    # Display the first few rows
    print(doc_topic_df.head())
    
    # Source for Above Code
    # Kapadia, S. (2024, April 15). Topic Modeling in Python: Latent Dirichlet Allocation (LDA). Medium. https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
    
    #Step 9: Retrieve the top 10 documents per topic
    
    #Extract the column containing document text
    documents = df["Text"].tolist()  # Ensure this column contains raw text data
    
    #Ensure 'documents' is correctly loaded
    print("Loaded", len(documents), "documents.")
    
    # Set the number of top documents to retrieve per topic
    top_n = 10 
    
    #Dictionary to store the top documents per topic
    top_docs_per_topic = {}
    
    #Count the number of unique dominant topics assigned to documents (should be 3, per final model)
    num_topics = doc_topic_df["Dominant_Topic"].nunique()
    
    for topic in range(num_topics):
        # Filter documents belonging to the current topic
        topic_docs = doc_topic_df[doc_topic_df["Dominant_Topic"] == topic].sort_values(by="Top_Probability", ascending=False)
    
        # Select the top N documents
        top_docs = topic_docs.head(top_n)["Document_Index"].tolist()
    
        # Store the actual text from the dataset
        top_docs_per_topic[topic] = [documents[i] for i in top_docs]
    
    # Export to a text file
    with open("top_documents_per_topic.txt", "w", encoding="utf-8") as f:
        for topic, docs in top_docs_per_topic.items():
            f.write("=== Topic " + str(topic) + " ===\n")
            for i, doc in enumerate(docs):
                f.write("\nSample " + str(i+1) + ":\n" + str(doc) + "\n")
            f.write("\n" + "="*50 + "\n")
    
    print("Export complete: top_documents_per_topic.txt")

if __name__ == "__main__":
    main()
