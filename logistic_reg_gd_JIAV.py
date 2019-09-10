from random import seed
from random import randrange
from csv import reader
from math import exp

# Carga un archivo CSV
def load_csv(filename):
	dataset = list()
	# Abre el archivo en modo lectura
	with open(filename, 'r') as file:
		# Lee el archivo y lo almacena en una variable
		csv_reader = reader(file)
		# Recorre cada linea del archivo
		for row in csv_reader:
			if not row:
				continue
			# Almacena en la lista dataset cada fila
			dataset.append(row)
	return dataset

def generate_csv(ids, predictions):
	# Crea el archivo results.csv y lo declara como file_csv para poder acceder a el
	with open("results.csv", "w") as file_csv:
		# En vez de imprimir en consola, imprime en file_csv
		print('PassengerId,Survived', file=file_csv)
		# Imprime el id del test sample y la predicción
		for i in range(len(predictions)):
			print('{},{}'.format(ids[i],predictions[i]), file=file_csv)

# Convierte los valores del CSV de strings a flotantes d
def str_column_to_float(dataset, test=False):
	ids = list()
	# i es el numero de columna
	for i in range(len(dataset[0])):
		# para cada fila va a convertir unicamente la columna indicada por i (recorre en vertical)
		for row in dataset:
			if test and i==0:
				ids.append(int(row[i].strip()))
			else:
			    row[i] = float(row[i].strip())
	if test:
		for row in dataset:
			row.pop(0)
		return ids

# Obtiene el valor maximo, el valor minimo y el promedio de cada columna para utilizarlos en la normalizacion
def dataset_minmaxavg(dataset):
	# minmaxavg es una lista donde se va a guardar el valor maximo, minimo y promedio de cada columna
	minmaxavg = list()
	for i in range(len(dataset[0])):
		# col_values almacena todos los valores de cada columna
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		value_avg = sum(col_values)/len(col_values)
		# Append agrega a la lista minmaxavg el valor minimo, el valor maximo y el valor promedio
		minmaxavg.append([value_min, value_max, value_avg])
	return minmaxavg

# Escala los valores del dataset en el rango 0 - 1
# https://kharshit.github.io/blog/2018/03/23/scaling-vs-normalization
def scale_dataset(dataset, minmaxavg):
	# Recorre de manera horizontal
	for row in dataset:
		for i in range(len(row)):
			# nuevo_valor = (valor_original - valor_min)/(valor_max - valor_min)
			row[i] = (row[i] - minmaxavg[i][0]) / (minmaxavg[i][1] - minmaxavg[i][0])
			# otra opcion podria ser normalizar 
			# nuevo_valor = (valor_original - valor_promedio)/(valor_max - valor_min)
			# row[i] = (row[i] - minmaxavg[i][2]) / (minmaxavg[i][1] - minmaxavg[i][0])
			# pero habria que modificar el range de la siguiente manera range(0, len(row)-1, 1) 
			# para que no modifique la ultima columna (los outputs ó Y)

# Separa el dataset en k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	# Crea una copia del dataset
	dataset_copy = list(dataset)
	# Fold_size = num_filas / n_folds
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			# Asigna un numero aleatorio dentro del numero de filas en el dataset_copy
			index = randrange(len(dataset_copy))
			# Agrega el elemento relacionado al index de la lista fold y los saca del dataset_copy
			fold.append(dataset_copy.pop(index))
		# Guarda en dataset_split cada uno de los folds que se generan en el for loop
		dataset_split.append(fold)
	return dataset_split

# Regresion lineal con stochatic gradient descent
def logistic_regression(train, test, l_rate, n_epoch):
	predictions = list()
	# coef recupera los coeficientes finales despues de entrenar
	coef = coefficients_sgd(train, l_rate, n_epoch)
	for row in test:
		yhat = predict(row, coef)
		yhat = round(yhat)
		predictions.append(yhat)
	print (coef)
	print ('\n')
	# Guarda los coeficientes en una lista para obtener el mejor al final
	coef_list.append(coef)
	return(predictions)

# Entrena el fold y regresa los coeficientes finales despues de n_epochs iteraciones
def coefficients_sgd(train, l_rate, n_epoch):
	# Inicializa en 0.0 los coeficientes
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			# Se hace una prediccion para cada fila
			yhat = predict(row, coef)
			# Se calcula el error restandole al valor esperado la prediccion
			error = row[-1] - yhat
			# se actualiza el valor del intercept(bias)
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
			# Se actualizan el resto de los coeficientes
			for i in range(len(row)-1):
				# new_coef = old_coef + (l_rate*error*prediction*(1-prediction)*Xi)
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
	return coef

# Genera las prediciones de cada fila
def predict(row, coefficients):
	# Guarda el valor del bias en yhat
	yhat = coefficients[0]
	# Recorre las columnas de la fila
	for i in range(len(row)-1):
		# Suma los resultados de la multiplicacion de coeficientes por los valores de Xi
		yhat += coefficients[i + 1] * row[i]
	# Regresa la sigmoide
	return 1.0 / (1.0 + exp(-yhat))

# Calcula el porcentaje de precision
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0


# Esta semilla ayuda a que cada que se corra el programa, los numeros "aleatorios" en 
# randrange de la funcion cross validation sean los mismos
seed(1)
# load and prepare data
filename = 'train.csv'
dataset = load_csv(filename)
str_column_to_float(dataset)
# normalize
minmaxavg = dataset_minmaxavg(dataset)
scale_dataset(dataset, minmaxavg)
# evaluate algorithm
n_folds = 5
l_rate = 0.06
n_epoch = 100
# Separa el dataset en n_folds semi-datasets
folds = cross_validation_split(dataset, n_folds)
scores = list()
coef_list = list()
# Itera en cada semi-dataset
for fold in folds:
	# Agrega todos los folds a train_set
	train_set = list(folds)
	# Quita el fold actual de train_set
	train_set.remove(fold)
	# Junta el resto de folds para tener un dataset completo
	train_set = sum(train_set, [])
	test_set = list()
	for row in fold:
		row_copy = list(row)
		# Agrega los valores del fold a test_set para con estos evaluar la precisión 
		test_set.append(row_copy)
		# Borra el label del fold para poder hacer pruebas
		row_copy[-1] = None
	# Con el train_set se corre la regresion logistica n_epoch iteraciones y con el test_set se evalua
	predicted = logistic_regression(train_set, test_set, l_rate, n_epoch)
	# Se guardan los valores esperados en actual
	actual = [row[-1] for row in fold]
	# Se mide la precision comparando los valores predichos con los valores esperados
	accuracy = accuracy_metric(actual, predicted)
	# Agrega la precision a la lista de scores
	scores.append(accuracy)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%\n\n' % (sum(scores)/float(len(scores))))

# Obtiene el score con mejor precision
best_score = scores.index(max(scores))
# Obtiene los coeficientes con los que se obtuvo el mejor score
best_coef = (coef_list[best_score])
print('Best Coefficients: \n%s\n' % best_coef)

# Test
# load and prepare data
test_filename = 'test.csv'
test_dataset = load_csv(test_filename)
ids = str_column_to_float(test_dataset, True)
# scale
minmax = dataset_minmaxavg(test_dataset)
scale_dataset(test_dataset, minmax)

predictions = list()
for row in test_dataset:
	yhat = predict(row, best_coef)
	yhat = round(yhat)
	predictions.append(yhat)

generate_csv(ids, predictions)