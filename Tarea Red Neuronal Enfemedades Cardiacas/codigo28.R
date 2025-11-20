# ============================
# Red neuronal para enfermedad cardíaca 
# ============================

library(dplyr)
library(neuralnet)

#  Leer y preparar datos
file_path <- "reprocessed.hungarian.data"
datos <- read.table(file_path, header = FALSE)

colnames(datos) <- c("age","sex","cp","trestbps","chol","fbs","restecg","thalach",
                     "exang","oldpeak","slope","ca","thal","num")

datos[datos == -9] <- NA

# Imputar NA con la media
datos <- datos %>% mutate(across(everything(), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# Crear variable binaria
datos$num_bin <- ifelse(datos$num == 0, 0, 1)
datos$num <- NULL

print(table(datos$num_bin))  # Comprobar balance de clases

#  Normalizar
normalize <- function(x) {
  if (length(unique(x)) == 1) return(rep(0, length(x)))
  (x - min(x)) / (max(x) - min(x))
}

datos_norm <- as.data.frame(lapply(datos[, -14], normalize))
datos_norm$num_bin <- datos$num_bin

#  Dividir en entrenamiento (70%) y prueba (30%)
set.seed(123)
train_idx <- sample(1:nrow(datos_norm), 0.7 * nrow(datos_norm))
train_data <- datos_norm[train_idx, ]
test_data  <- datos_norm[-train_idx, ]

# 4 Entrenar la red neuronal
f <- as.formula("num_bin ~ age + sex + cp + trestbps + chol + fbs + restecg + 
                 thalach + exang + oldpeak + slope + ca + thal")

set.seed(123)
nn <- neuralnet(f, data = train_data, hidden = 2,
                linear.output = FALSE, stepmax = 1e6)

# Verificar convergencia
if (is.null(nn$weights)) {
  stop("⚠️ El entrenamiento no convergió. Prueba con menos neuronas o más stepmax.")
}

# Graficar red
plot(nn)

# 5️⃣ Predicciones en test
pred <- compute(nn, test_data[, -14])
pred_prob <- pred$net.result
pred_class <- ifelse(pred_prob > 0.5, 1, 0)

# 6️⃣ Evaluar resultados
conf_matrix <- table(Predicted = pred_class, Actual = test_data$num_bin)
print(conf_matrix)

accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("Precisión de la red neuronal en TEST:", round(accuracy * 100, 2), "%\n")


