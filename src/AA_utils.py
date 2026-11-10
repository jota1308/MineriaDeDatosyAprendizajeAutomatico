import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay,PrecisionRecallDisplay

def mostrar_curva_roc(modelo, x_test, y_test):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve
    
    #  Obtener probabilidades de clase positiva
    y_scores = modelo.predict_proba(x_test)[:, 1]

    #  Calcular curvas
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_scores)
    prec, rec, thresholds_pr = precision_recall_curve(y_test, y_scores)

    #  Seleccionar algunos umbrales de ejemplo
    example_thresholds = [0.9, 0.5, 0.1]

    #  Graficar ROC y PR con puntos destacados
    plt.figure(figsize=(12,5))
    # --- Curva ROC ---
    plt.subplot(1,2,1)
    plt.plot(fpr, tpr, label="Curva ROC")
    for thr in example_thresholds:
        idx = (abs(thresholds_roc - thr)).argmin()
        plt.scatter(fpr[idx], tpr[idx], marker="o", s=60, 
                    label=f"Model threshold = {thr:.1f}")
    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    plt.xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
    plt.ylabel("Tasa de Verdaderos Positivos (Recall)")
    plt.title("Curva ROC con thresholds de modelo")
    plt.legend()

    # --- Curva Precision-Recall ---
    plt.subplot(1,2,2)
    plt.plot(rec, prec, label="Curva PR")
    for thr in example_thresholds:
        idx = (abs(thresholds_pr - thr)).argmin()
        plt.scatter(rec[idx], prec[idx], marker="o", s=60, 
                    label=f"Model threshold = {thr:.1f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision-Recall con thresholds de modelo")
    plt.legend()

    plt.tight_layout()
    plt.show()


def mostrar_matriz_confusion(Y_test, y_pred, classes=None):
    from sklearn.metrics import confusion_matrix
    import plotly.express as px
    import numpy as np
    
    cm = confusion_matrix(Y_test, y_pred)
    n_clases = cm.shape[0]
    
    # Si no nos pasan clases, o la longitud no coincide, generamos nombres genéricos
    if classes is None or len(classes) != n_clases:
        classes = [f"Clase {i}" for i in range(n_clases)]
        
    fig = px.imshow(cm,
                    labels=dict(x="Predicción", y="Real", color="Cantidad"),
                    x=classes,
                    y=classes,
                    text_auto=True,
                    color_continuous_scale='Blues')
    fig.update_layout(title="Matriz de Confusión")
    fig.show()

def plot_training_curves(history):
    # Plotting the training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid(True)
    
    plt.show()
    
def graficar_punto_elbow(X, N, GMM=False):
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    import plotly.graph_objects as go

    Nc = range(1, N+1)
    if (GMM):
        kmeans = [GaussianMixture(n_components=i) for i in Nc]
    else:
        kmeans = [KMeans(n_clusters=i) for i in Nc]

    if (GMM):
        score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
        scoreBIC = [kmeans[i].fit(X).bic(X) for i in range(len(kmeans))]

        fig_verosimilitud = go.Figure(data=go.Scatter(x=list(Nc), y=score))
        fig_verosimilitud.update_layout(title='Elbow Curve (log(verosimilitud))', xaxis_title='Número de Clusters', yaxis_title='log(verosimilitud)')
        fig_verosimilitud.show()

        fig_bic = go.Figure(data=go.Scatter(x=list(Nc), y=scoreBIC))
        fig_bic.update_layout(title='BIC Curve (menos es mejor)', xaxis_title='Número de Clusters', yaxis_title='Bayesian information criterion')
        fig_bic.show()
    else:
        score = [kmeans[i].fit(X).inertia_ for i in range(len(kmeans))]
        fig_elbow = go.Figure(data=go.Scatter(x=list(Nc), y=score))
        fig_elbow.update_layout(title='Elbow Curve (Promedio de sumas de distancias)', xaxis_title='Número de Clusters', yaxis_title='Promedio de sumas de distancias')
        fig_elbow.show()
    

# índice Silhouette
def graficar_indice_silhouette(X, N, GMM=False):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.mixture import  GaussianMixture
    import plotly.graph_objects as go

    plt.figure()
    Nc = range(2, N+1)
    if (GMM):
        # kmeans = [GaussianMixture(n_components=i) for i in Nc]
        print("Sin terminar silhouette para GMM")
    else:
        kmeans = [KMeans(n_clusters=i, n_init=10) for i in Nc] # Added n_init=10 to avoid warning

    # score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
    score = [silhouette_score(X, kmeans[i].fit(X).predict(X)) for i in range(len(kmeans))]

    fig_silhouette = go.Figure(data=go.Scatter(x=list(Nc), y=score))
    fig_silhouette.update_layout(title='Índice silhouette para distintos K (más es mejor)', xaxis_title='Número de Clusters', yaxis_title='Promedio de índice silhouette')
    fig_silhouette.show()
    
# grafico     
def graficar_indice_silhouette_k(X, n_clusters):
    from sklearn.metrics import silhouette_samples, silhouette_score
    from sklearn.cluster import KMeans
    import plotly.graph_objects as go
    import numpy as np
    import plotly.figure_factory as ff


    clusterer = KMeans(n_clusters=n_clusters, n_init=10) # Added n_init=10 to avoid warning
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    data = []
    group_labels = []
    colors = []
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        # Create a trace for each cluster
        trace = go.Scatter(
            x=ith_cluster_silhouette_values,
            y=np.arange(y_lower, y_upper),
            mode='lines',
            name=f'Cluster {i}',
            fill='tozerox'
        )
        data.append(trace)

        y_lower = y_upper + 10

    # Add average silhouette score line
    avg_line = go.Scatter(
        x=[silhouette_avg, silhouette_avg],
        y=[0, y_lower],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Average Silhouette Score'
    )
    data.append(avg_line)

    fig = go.Figure(data=data)
    fig.update_layout(
        title="Silhouette plot para cada Cluster.",
        xaxis_title="Coeficientes silhouette",
        yaxis_title="Cluster label",
        yaxis=dict(showticklabels=False),
        xaxis=dict(range=[-0.1, 1]),
        showlegend=True
    )
    fig.show()
 
    
def graficar_GMM(data, modelo, labels= False, probs= False):
    from matplotlib.colors import LogNorm
    import plotly.graph_objects as go
    import numpy as np

    # display predicted scores by the model as a contour plot
    cant_points= 200
    x = np.linspace(min(data[:, 0])*0.9, max(data[:, 0])*1.1, cant_points)
    y = np.linspace(min(data[:, 1])*0.9, max(data[:, 1])*1.1,cant_points)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T

    if (probs):
        Z = np.exp(modelo.score_samples(XX))
        title = 'Probabilidades de la GMM'
        colorscale = 'Plasma' # Change colorscale for probability plot
        # Increase the number of contour levels for better resolution
        contours_dict = dict(
            coloring='heatmap',
            showlabels=True, # show labels on contours
            labelfont=dict ( # label font properties
                size=12,
                color='white',
            )
        )
    else:
        Z = -modelo.score_samples(XX)
        title = 'Negative log-likelihood de la GMM'
        colorscale = 'Viridis'
        contours_dict = dict(
            coloring='heatmap',
            showlabels=True, # show labels on contours
            labelfont=dict ( # label font properties
                size=12,
                color='white',
            ),
            # Add more contour levels
            start=0,
            end=Z.max(),
            size=(Z.max() - Z.min()) / 20 # Adjust size to control number of levels
        )


    Z = Z.reshape(X.shape)

    fig = go.Figure(data = go.Contour(
        z=Z,
        x=x,
        y=y,
        colorscale=colorscale, # Use the selected colorscale
        contours=contours_dict # Use the defined contours dictionary
    ))

    if (labels):
        etiq= modelo.predict(data)
        fig.add_trace(go.Scatter(x=data[:, 0], y=data[:, 1], mode='markers', marker=dict(color=etiq, size=5), name='Data points'))
    else:
        fig.add_trace(go.Scatter(x=data[:, 0], y=data[:, 1], mode='markers', marker=dict(color='Red', size=5), name='Data points'))

    centros= modelo.means_
    fig.add_trace(go.Scatter(x=centros[:, 0], y=centros[:, 1], mode='markers', marker=dict(color='Blue', size=20, symbol='star'), name='Centroids'))

    fig.update_layout(
        title=title,
        xaxis_title='X',
        yaxis_title='Y',
        showlegend=True,  # Ensure legend is shown
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    fig.show()
    
def plot_DBSCAN_2d(data, dbscan_model,EPS, MINSAM, n_clusters):
  import plotly.graph_objects as go
  
  #% visualizar predicción
  labels= dbscan_model.labels_
  noise_maks = labels==-1
  core_samples_mask = np.zeros_like(dbscan_model.labels_, dtype=bool)
  core_samples_mask[dbscan_model.core_sample_indices_] = True

  # Create a Plotly figure
  fig = go.Figure()

  # plot RUIDO
  x_noise= data[noise_maks, 0] # datos RUIDO (outliers)
  y_noise= data[noise_maks, 1]
  fig.add_trace(go.Scattergl(
      x=x_noise,
      y=y_noise,
      mode='markers',
      marker=dict(
          color='rgba(0.2, 0.2, 0.2, 0.8)', # Dark grey with some transparency
          size=7, 
          symbol='x'
      ),
      name='Noise'
  ))

  # plot datos CORE en clusters
  x_core= data[~noise_maks & core_samples_mask, 0]
  y_core= data[~noise_maks & core_samples_mask, 1]
  labels_core = labels[~noise_maks & core_samples_mask]
  fig.add_trace(go.Scattergl(
      x=x_core,
      y=y_core,
      mode='markers',
      marker=dict(
          color=labels_core, # Color by cluster label
          size=10,
          opacity=0.8,
          colorscale='Viridis' 
      ),
      name='Cores'
  ))

  # plot datos NO CORE en clusters (Borders)
  x_border= data[~noise_maks & ~core_samples_mask, 0]
  y_border= data[~noise_maks & ~core_samples_mask, 1]
  labels_border = labels[~noise_maks & ~core_samples_mask]
  fig.add_trace(go.Scattergl(
      x=x_border,
      y=y_border,
      mode='markers',
      marker=dict(
          color=labels_border, # Color by cluster label
          size=7, 
          symbol='diamond', # Use a different symbol for borders
          opacity=0.6,
          colorscale='Viridis' # Use the same colorscale as cores
      ),
      name='Borders'
  ))

  title= "DBSCAN (eps= {}, min_samples= {}, #clusters= {})".format(EPS, MINSAM, n_clusters)
  fig.update_layout(
      title=title,
      xaxis_title='Feature 1', # Update axis titles if you know what they represent
      yaxis_title='Feature 2',
      hovermode='closest' # Shows the data point values on hover
  )

  fig.show()


def mostrar_frontera_decision(modelo, x_train, y_train, x_test, y_test, sigmoid=False):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from matplotlib.colors import ListedColormap

    # Unir datos
    X = np.vstack((x_train, x_test))
    Y = np.concatenate((y_train, y_test))
    clases = np.unique(Y)
    n_clases = len(clases)

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    pad_x = (x_max - x_min) * 0.01
    pad_y = (y_max - y_min) * 0.01

    x_limit_min, x_limit_max = x_min - pad_x, x_max + pad_x
    y_limit_min, y_limit_max = y_min - pad_y, y_max + pad_y

    xx, yy = np.meshgrid(np.arange(x_limit_min, x_limit_max, 0.02),
                         np.arange(y_limit_min, y_limit_max, 0.02))
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Paleta de colores dinámica
    if n_clases <= 2:
        colores = ['red', 'blue']
        cmap_background = ListedColormap(['#FFAAAA', '#AAAAFF'])
    else:
        try:
            cmap_base = plt.colormaps['tab10']
        except AttributeError:
            cmap_base = plt.cm.get_cmap('tab10')
        colores = [cmap_base(i) for i in range(n_clases)]
        cmap_background = ListedColormap(colores)

    # Graficar fondo
    if sigmoid and n_clases <= 2:
        if hasattr(modelo, "predict_proba"):
            Z = modelo.predict_proba(grid)[:, 1]
        else:
            Z = modelo.decision_function(grid)
            Z = 1 / (1 + np.exp(-Z))
        Z = Z.reshape(xx.shape)
        contour = ax.contourf(xx, yy, Z, levels=100, cmap='RdBu', alpha=0.6)
        fig.colorbar(contour, label='Probabilidad Clase 1')
    else:
        Z_raw = modelo.predict(grid)
        # Forzar un mapeo exacto a índices enteros y discretos para evitar saltos en el colormap
        Z_idx = np.zeros_like(Z_raw, dtype=int)
        for i, c in enumerate(clases):
            Z_idx[Z_raw == c] = i
        Z_idx = Z_idx.reshape(xx.shape)
        
        # Establecer levels fijos para que el colormap asigne 1 color puro a cada índice
        ax.contourf(xx, yy, Z_idx, levels=np.arange(n_clases + 1) - 0.5, cmap=cmap_background, alpha=0.4)

    # Graficar puntos dinámicamente según las clases
    handles_clases = []
    for i, c in enumerate(clases):
        color = colores[i]

        # Train
        ax.scatter(x_train[y_train == c, 0], x_train[y_train == c, 1],
                   c=[color], marker='o', edgecolors='k', s=50, alpha=0.8)
        # Test
        ax.scatter(x_test[y_test == c, 0], x_test[y_test == c, 1],
                   c=[color], marker='P', edgecolors='k', s=60, alpha=0.8)

        handle = mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                               markersize=8, label=f'Clase {int(c)}')
        handles_clases.append(handle)

    # Leyenda 1: Clases
    first_legend = ax.legend(handles=handles_clases, loc='upper right', title="Clases" if n_clases > 2 else None)
    ax.add_artist(first_legend)

    # Leyenda 2: Conjuntos
    train_marker = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=8, label='Entrenamiento')
    test_marker = mlines.Line2D([], [], color='black', marker='P', linestyle='None', markersize=8, label='Prueba')
    ax.legend(handles=[train_marker, test_marker], loc='lower left')

    ax.set_xlim(x_limit_min, x_limit_max)
    ax.set_ylim(y_limit_min, y_limit_max)
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')

    titulo = 'Frontera de Decisión'
    if sigmoid and n_clases <= 2:
        titulo += ' (Sigmoidea)'
    ax.set_title(titulo)
    plt.show()
