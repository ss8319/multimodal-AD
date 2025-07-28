import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import euclidean_distances
from typing import Dict, Any
import os
from datetime import datetime

# Try to import UMAP, but make it optional
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. UMAP visualization will be skipped.")

def find_latest_experiment(config: 'AutoencoderConfig', dataset_name: str) -> str:
    """Find the most recent experiment directory for a dataset"""
    dataset_path = os.path.join(config.base_path, dataset_name)
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset directory not found: {dataset_path}")
    
    experiments = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    if not experiments:
        raise ValueError(f"No experiment directories found in {dataset_path}")
    
    # Sort by creation time and use the most recent
    experiments.sort(key=lambda x: os.path.getctime(os.path.join(dataset_path, x)), reverse=True)
    return os.path.join(dataset_path, experiments[0])

def get_experiment_paths(config: 'AutoencoderConfig', dataset_name: str, experiment_dir: str = None) -> Dict[str, str]:
    """Get paths for a specific experiment"""
    if experiment_dir is None:
        experiment_dir = find_latest_experiment(config, dataset_name)
    
    experiment_name = os.path.basename(experiment_dir)
    return config.get_paths(dataset_name, experiment_name)

def load_visualization_data(dataset_config: 'DatasetConfig', paths: Dict[str, str]) -> Dict[str, Any]:
    """Load data for visualization"""
    train_features = np.load(paths['train_features'])
    test_features = np.load(paths['test_features'])
    
    # Load original data for labels
    df = pd.read_csv(dataset_config.metadata_path)
    y = (df['research_group'] == 'AD').astype(int)
    
    # Split labels to match train/test split
    from sklearn.model_selection import train_test_split
    _, _, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42, stratify=y)
    
    return {
        'train_features': train_features,
        'test_features': test_features,
        'y_train': y_train,
        'y_test': y_test
    }

def plot_feature_quality_analysis(data: Dict[str, Any], save_path: str):
    """Plot feature quality analysis"""
    train_features = data['train_features']
    test_features = data['test_features']
    
    plt.figure(figsize=(20, 5))
    
    # Feature distribution
    plt.subplot(1, 4, 1)
    plt.hist(train_features.flatten(), bins=50, alpha=0.7, density=True, label='Train')
    plt.hist(test_features.flatten(), bins=50, alpha=0.7, density=True, label='Test')
    plt.title('Feature Distribution')
    plt.xlabel('Feature Values')
    plt.ylabel('Density')
    plt.legend()
    
    # Feature variance
    plt.subplot(1, 4, 2)
    feature_vars = np.var(train_features, axis=0)
    plt.bar(range(1, len(feature_vars)+1), feature_vars)
    plt.title('Feature Variance')
    plt.xlabel('Feature Index')
    plt.ylabel('Variance')
    
    # Feature correlation
    plt.subplot(1, 4, 3)
    feature_corr = np.corrcoef(train_features.T)
    sns.heatmap(feature_corr, cmap='coolwarm', center=0, 
                xticklabels=range(1, train_features.shape[1]+1),
                yticklabels=range(1, train_features.shape[1]+1))
    plt.title('Feature Correlation')
    
    # Feature importance (LDA)
    plt.subplot(1, 4, 4)
    lda = LinearDiscriminantAnalysis()
    lda.fit(train_features, data['y_train'])
    feature_importance = np.abs(lda.coef_[0])
    plt.bar(range(1, len(feature_importance)+1), feature_importance)
    plt.title('Feature Importance (LDA)')
    plt.xlabel('Feature Index')
    plt.ylabel('|LDA Coefficient|')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_dimensionality_reduction(data: Dict[str, Any], save_path: str):
    """Plot dimensionality reduction visualizations"""
    train_features = data['train_features']
    test_features = data['test_features']
    y_train = data['y_train']
    y_test = data['y_test']
    
    plt.figure(figsize=(20, 10))
    
    # PCA
    pca = PCA(n_components=2)
    train_pca = pca.fit_transform(train_features)
    test_pca = pca.transform(test_features)
    
    plt.subplot(2, 3, 1)
    scatter = plt.scatter(train_pca[:, 0], train_pca[:, 1], c=y_train, cmap='viridis', alpha=0.7)
    plt.title(f'PCA (Train)\nExplained variance: {pca.explained_variance_ratio_.sum():.2%}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.colorbar(scatter, label='AD (1) vs CN (0)')
    
    plt.subplot(2, 3, 2)
    scatter = plt.scatter(test_pca[:, 0], test_pca[:, 1], c=y_test, cmap='viridis', alpha=0.7)
    plt.title('PCA (Test)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.colorbar(scatter, label='AD (1) vs CN (0)')
    
    # t-SNE
    if len(train_features) > 10:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(train_features)-1))
        train_tsne = tsne.fit_transform(train_features)
        
        plt.subplot(2, 3, 3)
        scatter = plt.scatter(train_tsne[:, 0], train_tsne[:, 1], c=y_train, cmap='viridis', alpha=0.7)
        plt.title('t-SNE (Train)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.colorbar(scatter, label='AD (1) vs CN (0)')
    
    # UMAP
    if len(train_features) > 10 and UMAP_AVAILABLE:
        try:
            reducer = umap.UMAP(random_state=42, n_neighbors=min(15, len(train_features)-1))
            train_umap = reducer.fit_transform(train_features)
            
            plt.subplot(2, 3, 4)
            scatter = plt.scatter(train_umap[:, 0], train_umap[:, 1], c=y_train, cmap='viridis', alpha=0.7)
            plt.title('UMAP (Train)')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.colorbar(scatter, label='AD (1) vs CN (0)')
        except Exception as e:
            print(f"Warning: UMAP visualization failed: {e}")
            plt.subplot(2, 3, 4)
            plt.text(0.5, 0.5, 'UMAP not available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('UMAP (Train)')
    elif len(train_features) > 10:
        plt.subplot(2, 3, 4)
        plt.text(0.5, 0.5, 'UMAP not available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('UMAP (Train)')
    
    # Clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(train_features)
    silhouette_avg = silhouette_score(train_features, cluster_labels)
    
    plt.subplot(2, 3, 5)
    scatter = plt.scatter(train_pca[:, 0], train_pca[:, 1], c=cluster_labels, cmap='Set1', alpha=0.7)
    plt.title(f'K-means Clustering\nSilhouette: {silhouette_avg:.3f}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.colorbar(scatter, label='Cluster')
    
    # Distance matrix
    plt.subplot(2, 3, 6)
    distances = euclidean_distances(train_features)
    plt.imshow(distances, cmap='viridis')
    plt.title('Sample Distance Matrix')
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')
    plt.colorbar(label='Euclidean Distance')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'train_pca': train_pca,
        'test_pca': test_pca,
        'cluster_labels': cluster_labels,
        'silhouette_score': silhouette_avg
    }

def analyze_features(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze feature quality and performance"""
    train_features = data['train_features']
    test_features = data['test_features']
    y_train = data['y_train']
    y_test = data['y_test']
    
    # KNN classification
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_features, y_train)
    y_pred = knn.predict(test_features)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Clustering quality
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(train_features)
    silhouette_avg = silhouette_score(train_features, cluster_labels)
    
    print("\n=== Feature Analysis Summary ===")
    print(f"Feature dimension: {train_features.shape[1]}")
    print(f"Training samples: {train_features.shape[0]}")
    print(f"Test samples: {test_features.shape[0]}")
    print(f"KNN accuracy: {accuracy:.3f}")
    print(f"Silhouette score: {silhouette_avg:.3f}")
    
    return {
        'knn_accuracy': accuracy,
        'silhouette_score': silhouette_avg,
        'feature_stats': {
            'mean': np.mean(train_features, axis=0),
            'std': np.std(train_features, axis=0),
            'min': np.min(train_features, axis=0),
            'max': np.max(train_features, axis=0)
        }
    }

def visualize_features_pipeline(config: 'AutoencoderConfig', dataset_config: 'DatasetConfig', 
                              experiment_dir: str = None) -> Dict[str, Any]:
    """Complete visualization pipeline"""
    # Get paths for the experiment
    paths = get_experiment_paths(config, dataset_config.name, experiment_dir)
    
    print(f"Visualizing features from experiment: {paths['experiment_dir']}")
    
    # Load data
    data = load_visualization_data(dataset_config, paths)
    
    # Create plots
    plot_feature_quality_analysis(data, paths['feature_quality'])
    dim_reduction_results = plot_dimensionality_reduction(data, paths['dim_reduction'])
    
    # Analyze features
    analysis_results = analyze_features(data)
    
    # Save visualization data
    visualization_data = {
        'train_features': data['train_features'],
        'test_features': data['test_features'],
        'y_train': data['y_train'],
        'y_test': data['y_test'],
        **dim_reduction_results,
        **analysis_results
    }
    
    np.save(paths['visualization_data'], visualization_data)
    
    print(f"Visualization completed for dataset: {dataset_config.name}")
    print(f"Results saved to: {paths['experiment_dir']}")
    return visualization_data
