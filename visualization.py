# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import signal
import numpy as np
from scipy.stats import ttest_ind
from sklearn.manifold import TSNE

def plot_pie_chart(data):
    emotion_counts = data['label'].value_counts()
    emotional_labels = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}
    emotion_labels = [emotional_labels[label] for label in emotion_counts.index]
    
    plt.figure(figsize=(8, 8))
    plt.pie(emotion_counts, labels=emotion_labels, autopct='%1.1f%%', startangle=140, colors=['red', 'yellow', 'green'])
    plt.title("Distribution of Emotions (0: NEGATIVE, 1: NEUTRAL, 2: POSITIVE)")
    plt.axis('equal')
    plt.show()

def plot_time_series(data, index):
    sample = data.loc[index, 'fft_0_b':'fft_749_b']
    plt.figure(figsize=(16, 10))
    plt.plot(range(len(sample)), sample)
    plt.title("EEG Time-Series Data")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()

def plot_power_spectral_density(sample, sampling_rate):
    frequencies, power_density = signal.welch(sample, fs=sampling_rate)
    plt.figure(figsize=(10, 6))
    plt.semilogy(frequencies, power_density)
    plt.title("Power Spectral Density")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency")
    plt.grid()
    plt.show()

def plot_correlation_heatmap(data):
    correlation_matrix = data.drop('label', axis=1).corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.show()

def plot_significant_non_significant_features(data):
    emotions = data['label'].unique()
    num_features = {emotion: {'significant': 0, 'non-significant': 0} for emotion in emotions}

    # Perform t-tests
    for emotion in emotions:
        subset = data[data['label'] == emotion]
        for feature in data.columns[:-1]:
            _, p_value = ttest_ind(subset[feature], data[feature])
            if p_value < 0.05:
                num_features[emotion]['significant'] += 1
            else:
                num_features[emotion]['non-significant'] += 1

    # Extract feature counts
    emotion_labels = list(num_features.keys())
    significant_counts = [num_features[emotion]['significant'] for emotion in emotion_labels]
    non_significant_counts = [num_features[emotion]['non-significant'] for emotion in emotion_labels]

    # Bar chart visualization
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(len(emotion_labels))

    plt.bar(index, significant_counts, bar_width, label='Significant', color='green')
    plt.bar(index + bar_width, non_significant_counts, bar_width, label='Non-Significant', color='red')

    plt.xlabel('Emotion (0: Negative, 1: Neutral, 2: Positive)')
    plt.ylabel('Number of Features')
    plt.title('Significant and Non-Significant Features by Emotion')
    plt.xticks(index + bar_width / 2, emotion_labels)
    plt.legend()

    # Display counts above bars
    for i, (significant_count, non_significant_count) in enumerate(zip(significant_counts, non_significant_counts)):
        plt.text(i, significant_count + non_significant_count + 1, f'S: {significant_count}\nNS: {non_significant_count}', ha='center', va='bottom', fontsize=12)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_tsne(data):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(data.drop('label', axis=1))
    tsne_df = pd.DataFrame(tsne_results, columns=['Dimension 1', 'Dimension 2'])
    tsne_df['label'] = data['label']

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Dimension 1', y='Dimension 2', hue='label', data=tsne_df, palette='viridis')
    plt.title("t-SNE Visualization")
    plt.show()
