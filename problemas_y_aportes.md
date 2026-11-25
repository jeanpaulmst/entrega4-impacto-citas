# Problemas y aportes de los modelos

Proyecto Integrador · Ciencia de Datos · UTN FRM · Grupo 10
¿Se puede anticipar, el día que se publica, cuánto se va a citar un artículo de Ciencia de Materiales?

---

## Problemas que encontramos

### 1. No ve venir a los éxitos

Es el problema central del regresor (Gradient Boosting):

- **Comprime los extremos.** A los artículos con más de 100 citas les estima 54 cuando la mediana real es 169. A los que no tienen ninguna cita les asigna ~0,9.
- **El error típico es un factor de ×2.** A un artículo con 20 citas lo puede estimar en 10 o en 40.
- **La causa no es el modelo sino los datos.** Las columnas del día de publicación dicen que un artículo «va a andar bien». Lo que lo convierte en un éxito es el hallazgo, y eso no está en ninguna columna. Aun con revista muy citada y bibliografía larga, el 28 % no llega al tercio alto.

### 2. Se apoya en conocer la revista

- Las dos columnas que más pesan son el prestigio previo de la revista y las referencias.
- Cuando la validación se arma por revista, el modelo se evalúa con revistas que no vio, y **la ventaja de Boosting sobre Ridge se reduce a la mitad** (de 0,020 a 0,010).
- Con una revista nueva, o sin historial (el 1 % de los artículos), el modelo pierde su mejor pista.

### 3. El tercio del medio es la frontera

En los extremos casi no confunde: solo el 3 % salta de «poco» a «muy citado» o al revés. Pero cuando dice «medianamente citado» acierta apenas la mitad, porque el tercio del medio limita con los otros dos.

### 4. El top 10 % es una clase chica

- Solo el **12 %** de los artículos es top 10 %. Con el umbral por defecto (0,5), la logística detectaba apenas el 28 % de ellos.
- Bajamos el umbral a **0,31**. Aun así, en el test encuentra **120 de los 247** top 10 % (la mitad) y da **125 falsas alarmas**: la mitad de los candidatos no se destaca.

### 5. Algunos modelos sobreajustaban

El sobreajuste es cuando el modelo memoriza el entrenamiento en lugar de aprender algo que sirva para casos nuevos.

- El **árbol de decisión sin límite** tenía error 0,000 en entrenamiento y 0,952 en validación.
- **Random forest** también tenía una brecha grande (0,256 contra 0,692).

Por eso se ajustaron los hiperparámetros con búsqueda en grilla y validación cruzada.

### 6. Riesgo de fuga

Muchas columnas de OpenAlex se calculan **con las citas del propio artículo**, por ejemplo `fwci` y los percentiles. Una de ellas reproduce `is_top10_percent` en el 100 % de las filas. Si no las sacábamos, el modelo «acertaba» copiando la respuesta. También se sacaron las columnas que crecen después de publicar.

### 7. Una población muy acotada

- Un solo campo (Ciencia de Materiales) y un solo año (2019). No sabemos si se sostiene en otros campos o años.
- La etiqueta top 10 % la calcula OpenAlex contra todo el subcampo, no contra nuestra muestra. Por eso la tasa base es 12 % y no 10 %.

### 8. El mejor modelo no se puede llevar a la página

Boosting no se puede escribir como fórmula, así que «Probá» estima las citas con Ridge. Ridge es peor por más de dos desvíos (MAE 0,709 contra 0,685 ± 0,011): queda por fuera de lo que permite la consigna, que pide un modelo a menos de un desvío del mejor.

### 9. Problemas de uso, más allá de lo técnico

- **Puede reforzar la desigualdad.** El modelo premia a las revistas que ya eran prestigiosas. Usado para decidir qué difundir, les daría todavía más visibilidad: es el «efecto Mateo», donde al que tiene se le da más. Y castiga a revistas nuevas o instituciones chicas.
- **Correlación no es causa.** Que más referencias vayan con más citas no quiere decir que agregar referencias aumente el impacto. Si alguien usara el modelo para decidir, se podría hacer trampa inflando la bibliografía.

---

## Qué aportes puede tener en un caso real

### 1. Triage: por dónde empezar a leer

Es el uso más claro. Ante cientos de artículos nuevos, ordenarlos por la probabilidad que da el modelo:

- En el **10 %** que el modelo ve más prometedor, el **53 %** fue top 10 %. Elegir al azar da 12 %.
- En la **mitad de abajo**, solo el **1,2 %** se destacó.

Le sirve a una editorial que elige qué difundir en el boletín del mes, a un comité que revisa trabajos, o a una biblioteca que arma alertas. No decide por nadie: ordena por dónde empezar.

### 2. Recomendar artículos nuevos que todavía no tienen citas

Los buscadores y recomendadores académicos suelen ordenar por citas, y un artículo recién publicado tiene cero. El modelo da una estimación de impacto el día uno. Sirve para no enterrar los trabajos nuevos mientras las citas llegan.

### 3. Un punto de referencia: «citas esperadas»

Funciona como los **goles esperados (xG)** en el fútbol: comparar lo que pasó con lo que se esperaba. La exploración ya permite ordenar por «más citas de las esperadas» y encontrar artículos que superaron su punto de partida. Ahí hay hallazgos que no se explican por la revista ni por el equipo, que es justo lo que más interesa descubrir. También sirve para comparar grupos de investigación de forma más justa: no contra las citas en crudo, sino contra lo esperable para revistas y equipos parecidos.

### 4. Lo que aprendimos de los datos, aunque no se use el modelo

- La revista y la bibliografía anticipan buena parte del impacto, y **se suman** entre sí.
- El mes de publicación **no importa**, y el acceso abierto importa solo según el tipo de revista.

Para un autor que decide dónde publicar, eso ya es información útil.

### Para qué no usarlo

Ni para **evaluar investigadores ni decidir contrataciones o fondos**. El modelo no lee el contenido, se equivoca justamente en los trabajos excepcionales, y castigaría a quien publica en revistas menos conocidas. Esto también lo plantean declaraciones como DORA, que piden no juzgar un trabajo por la revista donde sale.

### Qué haría falta para usarlo de verdad

- Reentrenarlo con **más campos y años**, y revisar cada tanto que siga funcionando, porque las costumbres de citación cambian.
- Sumar **el texto del artículo** (título y abstract), que es la pista más clara sobre el contenido.
- Medir si trata peor a revistas nuevas o instituciones chicas, y corregirlo.

---

**En una línea:** no predice el éxito, pero dice con bastante seguridad por dónde conviene empezar a mirar, y cuáles casi nunca se destacan.
