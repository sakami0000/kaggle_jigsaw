from pathlib import Path

TOXICITY_COLUMN = 'target'
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
AUX_TOXICITY_COLUMNS = [
    'severe_toxicity', 'obscene', 'identity_attack',
    'insult', 'threat', 'sexual_explicit']

OLD_TOXICITY_COLUMN = 'toxic'
OLD_IDENTITY_COLUMNS = [
    'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
OLD_AUX_TOXICITY_COLUMNS = [
    'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

EMBEDDING_FASTTEXT = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
EMBEDDING_GLOVE = '../input/glove840b300dtxt/glove.840B.300d.txt'

DATA_DIR = Path('../input/jigsaw-unintended-bias-in-toxicity-classification/')
TRAIN_DATA = DATA_DIR / 'train.csv'
TEST_DATA = DATA_DIR / 'test.csv'
SAMPLE_SUBMISSION = DATA_DIR / 'sample_submission.csv'

OLD_DIR = Path('../input/jigsaw-toxic-comment-classification-challenge/')
TRAIN_OLD = OLD_DIR / 'train.csv'
TEST_OLD = OLD_DIR / 'test.csv'
SAMPLE_OLD = OLD_DIR / 'sample_submission.csv'
