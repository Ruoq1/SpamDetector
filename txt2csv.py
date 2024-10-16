import os
import pandas as pd

# def enron dataset path, each dataset contains spam and ham folders
enron_datasets = [
    "/Users/diwakar/Code/enron/enron1",
    "/Users/diwakar/Code/enron/enron2",
    "/Users/diwakar/Code/enron/enron3",
    "/Users/diwakar/Code/enron/enron4",
    "/Users/diwakar/Code/enron/enron5",
    "/Users/diwakar/Code/enron/enron6"
]

# define a function to load emails from a folder
def load_emails(folder_path, label):
    emails = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):  # only load txt files
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='latin1') as file:
                content = file.read()
                emails.append({'content': content, 'label': label})  # add label
    return emails

# save all emails
all_emails = []

# traverse all enron spam and ham folders
for enron_dataset in enron_datasets:
    spam_folder = os.path.join(enron_dataset, 'spam')  # spam folder path
    ham_folder = os.path.join(enron_dataset, 'ham')    # ham folder path

    # load each enron dataset
    spam_emails = load_emails(spam_folder, 1)  # spam label 1
    ham_emails = load_emails(ham_folder, 0)    # ham label 0

    # combine all emails
    all_emails.extend(spam_emails + ham_emails)

# convert to DataFrame
df = pd.DataFrame(all_emails)

# save as csv
df.to_csv('spam_ham_dataset.csv', index=False)

print("All datasets saved as spam_ham_dataset.csv")
