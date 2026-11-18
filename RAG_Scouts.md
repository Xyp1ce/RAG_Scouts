# Contexto
La Asociación de Scouts de México (ASMAC) es un movimiento sin fines de lucro el cual tiene como objetivo el continuo crecimiento de los jóvenes scouts en el rango de edad de 6 a 22 años. Para lograr este continuo crecimiento el movimiento ofrece recompensar a aquellos Scouts que realicen una actividad de su interés y que promueva su crecimiento, dicha recompensa es una insignia el cual el Scout podrá portar en su uniforme llamadas ‘especialidades’. Dichas especialidades están segmentadas en diferentes áreas en las cuales el Scouts puede desarrollarse. 

El movimiento llegó a México en 1912 pero no fue hasta 1926 que la ASMAC fue reconocida a nivel mundial. En estos 100 años el movimiento ha pasado por muchos cambios administrativos y dichos cambios trajeron consigo cambios enormes en la logística del movimiento.

A partir del año 2026 se implementó un cambio a nivel nacional el cual cambió por completo la logística del movimiento, pero uno de los principales cambios fueron las categorías en las cuales fueron divididas las especialidades, pasando de unos pares a una gran variedad de áreas, esto se realizó con el fin de poder ser más específicos con el área en el que se trabaja.
Problemática
Cada cambio realizado en el movimiento trae consigo una ola de inconformidades y problemas a tratar, esto se intenta mitigar con cursos impartidos por gente capacitada para aquellos adultos que están frente a los jóvenes Scouts. Pero dichos cursos no siempre son comprendidos en su totalidad o intentan condensar toda la información en una plática de un día, lo cual genera que se omitan ciertos temas o no se le ponga tanto énfasis en otros. 

El cambio realizado en las especialidades fue algo que retrasó a muchas personas, al incrementar la cantidad de áreas posibles en las cuales trabajar y el que los adultos a cargo de los Scouts no terminen de comprender estos cambios, se ralentiza el inicio de la documentación necesaria para que los jóvenes puedan obtener su insignia, esto desfavorece al movimiento debido a que las especialidades son una de los principales actividades que el Scout puede realizar para su continuo crecimiento.

# Propuesta
Se propone la implementación de una GUI que se asimile a un chat de Gemini, Claude o ChatGPT que implemente un sistema RAG el cual funcionará como la base de conocimiento para el LLM. Dentro de la interfaz se podrán realizar preguntar sobre a qué área podría corresponder una acción la cual el Scout desee realizar y junto con el LLM se generará una respuesta cuyos fundamentos sean la información dentro del sistema RAG. 
Tecnologías 

## LLM (Large Language Model)
Un LLM o Modelo de Lenguaje Grande es un sistema de IA el cual fue pre entrenado con datos masivos que pueden entender y generar lenguaje natural, o texto que se asimile a uno generado por el humano. Los LLM pueden generar texto de manera automática con base a la información con la que fue entrenada y su aplicación entra en una gran variedad de trabajos y tareas.

La utilización de un LLM se ve justificada por la necesidad de tener una respuesta textual la cual pueda ser entendida por la persona que utilice el sistema. Un LLM genera texto similar al generado por un humano lo cual facilitará la explicación de la decisión tomada por el modelo de IA.

## Sistemas RAG
Los sistemas RAG (Retrieval-Augmented Generation o Generación Aumentada por Recuperación) son una arquitectura de Inteligencia Artificial que combina un modelo de lenguaje (LLM) con una base de datos externa de conocimiento. 

Los LLM tienden a alucinar o basarse en información obsoleta. Al contar con un esquema estructurado, un sistema RG soluciona el problema de las alucinaciones porque obliga a la IA a consultar únicamente la documentación oficial proporcionada antes de realizar una respuesta, lo que garantiza una mayor precisión.

# Metodología
Se comenzará con la captura de la información en manuales, esquemas y lineamientos del nuevo sistema de especialidades de la ASMAC 2026 para generar la base de conocimiento del sistema RAG, se deberá limpiar y organizar los textos normativos para que sirvan como contexto, y se deberán convertir las reglas en embeddings para almacenarlos en la base de datos del sistema RAG. 

Se desarrollará la Interfaz Gráfica de Usuario (GUI) en la cual el usuario podrá interactuar con el sistema, la interfaz tendrá un estilo visual similar a un chat de Inteligencia Artificial para evitar comportamientos no deseados.

Una vez generada la base, se integrará un LLM el cual será el encargado de procesar las consultas del usuario y generar respuestas en base al contexto dado por el sistema RAG. De esta manera, cuando el usuario ingrese una palabra o acción el sistema consultará la base de conocimiento para generar una respuesta justificada. 

Para las variables sintéticas se necesitan definir las características de las acciones y proyectos que un Scout propondrá de manera típica. Con base a estos datos se creará una base de datos sintética masiva que relacione distintas formas de describir una acción con su área de desempeño correspondiente.

# Resultados esperados
Como resultado de la implementación de un sistema RAG integrado con un modelo de lenguaje grande (LLM), se espera desarrollar una herramienta capaz de identificar y clasificar automáticamente las palabras o acciones ingresadas por el usuario, relacionándolas con las áreas de desempeño correspondientes. Con esto, se busca facilitar la búsqueda de información, proporcionar respuestas relevantes y reducir el tiempo que el usuario necesita para encontrar la información que desea.
