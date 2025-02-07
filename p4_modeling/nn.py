# Red neuronal
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l2 
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier
import joblib
from joblib import parallel_backend

class TrainNeuralNetwork():
    
    def __init__(self) -> None:
        pass

    def create_neural_network(self, input_shape, output_shape, activation='relu', hidden_layer_sizes=[50, 100], 
                          optimizer='adam', learning_rate=0.001, kernel_regularizer=0.001, 
                          batch_normalization=False, dropout_rate=None, metrics=['accuracy'],  # ['accuracy', Precision(), Recall()], 
                          verbose: int = 0):
        """
        Creación de la arquitectura de la red neuronal y del modelo.

        # Parameters
            input_shape: Número de neuronas en la capa de entrada, igual al número de variables predictoras (int).
            output_shape: Número de neuronas en la capa de salida, generalmente 1 para regresión o igual al número de clases en clasificación (int).
            activation: Función de activación para las capas ocultas (str, default='relu').
                Ejemplos: 'relu', 'sigmoid', 'tanh', 'softmax', etc.
            hidden_layer_sizes: Lista que define la cantidad de neuronas en cada capa oculta (list).
                Ejemplo: [50, 100] crea dos capas ocultas con 50 y 100 neuronas respectivamente.
            optimizer: Optimizador utilizado para entrenar la red (str o keras.optimizers).
                Ejemplo: 'adam', 'sgd', 'rmsprop', o un objeto de optimizador de Keras.
            learning_rate: Tasa de aprendizaje para el optimizador (float, default=0.001).
            kernel_regularizer: Coeficiente de regularización L2 para reducir el sobreajuste (float, default=0.001).
            batch_normalization: Si es True, añade capas de normalización por lotes después de cada capa oculta (bool, default=False).
            dropout_rate: Proporción de neuronas a desactivar en cada capa durante el entrenamiento para evitar sobreajuste (float, default=None).
                Ejemplo: dropout_rate=0.5 mantendrá el 50% de las neuronas activas en cada paso.
            metrics: Lista de métricas para evaluar el rendimiento del modelo (list, default=['accuracy', Precision(), Recall()]).

        # Return
            model: El modelo de red neuronal compilado y listo para entrenarse.
        """
        model = Sequential()
        if verbose >= 1:
            logger.info(f"Parametros a probar: {input_shape} {output_shape} {activation} {hidden_layer_sizes} {optimizer} {learning_rate} {kernel_regularizer} {batch_normalization} {dropout_rate} {metrics}")
        
        # First layer
        model.add(Input(shape=(input_shape,)))
  
        # Por Hidden layers
        for neurons in hidden_layer_sizes:
            model.add(Dense(neurons, activation=activation, kernel_regularizer=l2(kernel_regularizer)))
            
            # Dropout opcional en capas ocultas
            if dropout_rate:
                model.add(Dropout(dropout_rate))
            
            if batch_normalization:
                model.add(BatchNormalization())

        # Output layer (Capa de salida con softmax para clasificación multiclase)
        model.add(Dense(output_shape, activation='softmax'))

        # Configuración del optimizador
        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Optimizer '{optimizer}' not supported")

        # Compilación del modelo
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=metrics)
        
        return model
     
    def select_best_arquitecture(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32, verbose: int = 0):
        """
        Entrena múltiples arquitecturas de redes neuronales y selecciona la mejor según su desempeño en los datos de validación.

        # Parameters
            X_train: Conjunto de entrenamiento para las variables predictoras (DataFrame o array).
            y_train: Conjunto de entrenamiento para la variable respuesta (DataFrame o array).
            X_val: Conjunto de validación para las variables predictoras, usado para evaluar el rendimiento (DataFrame o array).
            y_val: Conjunto de validación para la variable respuesta (DataFrame o array).
            epochs: Número de épocas para entrenar cada arquitectura de red (int, default=20).
            batch_size: Tamaño de los lotes para el entrenamiento (int, default=32).
            verbose: Nivel de detalle de la salida durante el entrenamiento; 0 = silencioso, 1 = detallado, 2 = una barra de progreso por época (int, default=0).

        # Return
            best_model: Modelo de red neuronal con el mejor rendimiento en los datos de validación.
            best_params: Diccionario con los parámetros de la arquitectura seleccionada.
            best_score: Mejor puntaje obtenido en los datos de validación (ej. accuracy, F1, etc., dependiendo de la métrica definida).
        """
        start = time.time()

        # Lista para guardar los resultados y modelos
        results = []

        # Hiperparametros de arquitectura
        param_grid = { 
            'hidden_layer_sizes': [[50], [64, 32], [128, 64], [128, 64, 32]], # , [100, 50], [64, 32], [100, 100], [256, 128, 64], [1024, 512, 256],  [512, 256, 128, 64] (no gana y encima creo que es la causa del kill...)
            'learning_rate': [0.01, 0.1],
            'activation': ['relu', 'tanh'],
            'optimizer': ['adam'],
            'kernel_regularizer': [None, 0.01],
            'batch_normalization': [False],
            'dropout_rate': [0.2, None]
        }

        # Generar combinaciones de parámetros automáticamente
        param_combinations = list(product(*param_grid.values()))

        # Convertir etiquetas a formato one-hot --> Evita error target y output con different shape. 
        input_shape, output_shape = X_train.shape[1], 3  # Estaria bueno que sea automatico
        patience = int(epochs * 0.25)  # Por ejemplo, 20% de las épocas totales
        y_val_categorical = to_categorical(y_val, num_classes=output_shape)
        y_train_categorical = to_categorical(y_train, num_classes=output_shape)

        # Definir un callback de EarlyStopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        # timeout_callback = TimeoutCallback(max_seconds=100)  # Definir el límite de tiempo (por ejemplo, 300 segundos) # No se si funciona y tampoco creo que esta sea la causa de que tarde mucho tiempo.

        # Iterar sobre cada combinación
        for params in param_combinations:
            
            # Emparejar cada parámetro con su nombre desde `param_grid`
            param_dict = dict(zip(param_grid.keys(), params))
            # logger.warning(param_dict)

            # Creo red neuronal
            model = self.create_neural_network(
                input_shape=input_shape,
                output_shape=output_shape,
                **param_dict  # Desempaqueta el diccionario como argumentos nombrados
            )

            # Entrenar el modelo (usa el validation como test en vez de hacer cross val entre X_train)
            joblib.parallel.DEFAULT_BACKEND = "loky"
            with parallel_backend('threading'):
                history = model.fit(X_train, y_train_categorical, validation_data=(X_val, y_val_categorical), epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[early_stopping]) # timeout_callback

            # Evaluar en el set de validación
            val_loss, val_acc, val_precision, val_recall = model.evaluate(X_val, y_val_categorical, verbose=verbose)
            val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0  # Accuracy NO.

            results.append({
                'params': params,
                **param_dict,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1, 
                'model': model,
                # 'history': history # &lt;keras.src.callbacks.history.History object at 0x34f4a3e30&gt;
            })

        # Calculo metrica combinada entre f1_score y val_loss
        results = combined_metric(results)

        # Buscar la mejor combinación de hiperparámetros según la métrica (por ejemplo, accuracy)
        best_result = min(results, key=lambda x: x['combined_metric'])  # best_result = min(results, key=lambda x: x['val_loss'])  # best_result = max(results, key=lambda x: x['val_f1']) 

        # Imprimo rdos
        end = time.time()

        if verbose >= 0:
            logger.info(f"Best arquitecture: {best_result['params']}")
            logger.info(f"Metrics: Val loss: {best_result['val_loss']}  Val Accuracy: {best_result['val_acc']} Val f1: {best_result['val_f1']}")
            logger.info(f"\tSeleccion de hiperparametros optimos en {(end - start) / 60:.1f} minutos")

        return best_result['model'], best_result['params'], best_result['val_acc'], pd.DataFrame(results)
    
class TimeoutCallback(tf.keras.callbacks.Callback):
    def __init__(self, max_seconds):
        super(TimeoutCallback, self).__init__()
        self.max_seconds = max_seconds
        self.start_time = None

    def on_train_begin(self, logs=None):
        # Registrar el tiempo de inicio del entrenamiento
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        # Verificar el tiempo transcurrido
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_seconds:
            print(f'\nEntrenamiento detenido: tiempo máximo de {self.max_seconds} segundos alcanzado.')
            self.model.stop_training = True

