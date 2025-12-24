from models import train_and_save_models

if __name__ == '__main__':
    print('Training and saving logistic, svm, naive_bayes models...')
    saved = train_and_save_models(csv_path='Data/csvfile.csv', output_prefix='lead_classifier_')
    print('Saved:', saved)
