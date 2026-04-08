import logging
from google.cloud import firestore
from google.cloud.exceptions import GoogleCloudError
from google import genai
from typing import Optional, Union
from google.genai.types import EmbedContentConfig
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
import os
import hashlib
from datetime import datetime


# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FirestoreVectorStore:
    # Constante para el campo de embedding (fijo)
    EMBEDDING_KEY = "embedding"
    MAX_DIMENSION = 2048

    
    def __init__(self, project: str, database: str, collection: str, location: str = "us-east1"):
        """
        Inicializa el vector store de Firestore.
        
        Args:
            project: ID del proyecto de GCP
            database: Nombre de la base de datos de Firestore
            collection: Nombre de la colección donde se almacenarán los vectores
            location: Región de Google Cloud (default: "us-east1")
        """
        
        self.project = project
        self.collection = collection
        self.location = location
        
        # Configurar variables de entorno (con warnings si se sobrescriben)
        existing_project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if existing_project and existing_project != project:
            logger.warning(f"GOOGLE_CLOUD_PROJECT ya establecido como '{existing_project}', sobrescribiendo con '{project}'")
        
        existing_location = os.environ.get("GOOGLE_CLOUD_LOCATION")
        if existing_location and existing_location != location:
            logger.warning(f"GOOGLE_CLOUD_LOCATION ya establecido como '{existing_location}', sobrescribiendo con '{location}'")
        
        os.environ["GOOGLE_CLOUD_PROJECT"] = project
        os.environ["GOOGLE_CLOUD_LOCATION"] = location
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

        try:
            self.db = firestore.Client(project=project, database=database)
            self.genai_client = genai.Client(project=project, vertexai=True)
            logger.info(f"FirestoreVectorStore inicializado: proyecto={project}, database={database}, colección={collection}, location={location}")
        except GoogleCloudError as e:
            logger.error(f"Error al inicializar Firestore: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado al inicializar: {e}")
            raise
    
    def _validate_dimension(self, dimension: Optional[int]) -> None:
        """
        Valida que la dimensión no exceda el máximo permitido.
        
        Args:
            dimension: Dimensión a validar
            
        Raises:
            ValueError: Si la dimensión excede el máximo permitido
        """
        if dimension is not None and dimension > self.MAX_DIMENSION:
            error_msg = f"La dimensión {dimension} excede el máximo permitido de {self.MAX_DIMENSION}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _generate_document_id(self, doc: dict, text_key: str) -> str:
        """
        Genera un ID único (hash) para un documento basado en su contenido.
        
        Args:
            doc: Documento del cual generar el ID
            text_key: Clave del campo de texto principal
            
        Returns:
            Hash SHA256 del contenido del documento (16 primeros caracteres)
        """
        # Usar el texto principal para generar el hash
        content = str(doc.get(text_key, ""))
        # Agregar otros campos relevantes para mayor unicidad
        for key, value in sorted(doc.items()):
            if key not in [self.EMBEDDING_KEY, 'id', 'created_at', 'updated_at']:
                content += f"{key}:{value}"
        
        # Generar hash SHA256 y tomar los primeros 16 caracteres
        hash_object = hashlib.sha256(content.encode('utf-8'))
        return hash_object.hexdigest()[:16]
    
    def embed_texts(self, texts: list[str], embedding_model: str, dimension: Optional[int] = None):
        """
        Genera embeddings para una lista de textos.
        
        Args:
            texts: Lista de textos a embedear
            embedding_model: Modelo de embedding a usar
            dimension: Dimensión del embedding (opcional, máximo 2048)
            
        Returns:
            Lista de objetos Vector con los embeddings
            
        Raises:
            ValueError: Si la dimensión excede el máximo permitido
        """
        try:
            # Validar dimensión
            self._validate_dimension(dimension)
            
            # Límite de batch para la API de embeddings (máximo 250)
            MAX_BATCH_SIZE = 250
            
            logger.info(f"Generando embeddings para {len(texts)} textos con modelo {embedding_model}")
            
            all_vectors = []
            total_batches = (len(texts) + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
            
            # Procesar en lotes de máximo 250
            for i in range(0, len(texts), MAX_BATCH_SIZE):
                batch_texts = texts[i:i + MAX_BATCH_SIZE]
                batch_num = (i // MAX_BATCH_SIZE) + 1
                
                logger.info(f"Procesando lote {batch_num}/{total_batches} ({len(batch_texts)} textos)")
                
                embeddings = self.genai_client.models.embed_content(
                    model=embedding_model,
                    contents=batch_texts,
                    config=EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=dimension
                    )
                )
                
                # Validar dimensión resultante
                if embeddings.embeddings:
                    actual_dimension = len(embeddings.embeddings[0].values)
                    if actual_dimension > self.MAX_DIMENSION:
                        error_msg = f"La dimensión resultante {actual_dimension} excede el máximo permitido de {self.MAX_DIMENSION}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                
                # Pasar a clase vector y agregar a la lista total
                batch_vectors = [Vector(value=embedding.values) for embedding in embeddings.embeddings]
                all_vectors.extend(batch_vectors)
                logger.info(f"Lote {batch_num}/{total_batches} completado: {len(batch_vectors)} vectores generados")
            
            logger.info(f"Embeddings generados exitosamente: {len(all_vectors)} vectores en total")
            return all_vectors
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error al generar embeddings: {e}")
            raise
    
    def embed_documents(self, documents: list[dict], embedding_model: str = "gemini-embedding-001", dimension: int = 2048, text_key: str = "text"):
        """
        Genera embeddings para una lista de documentos y los agrega al campo fijo 'embedding'.
        
        Args:
            documents: Lista de diccionarios con los documentos
            embedding_model: Modelo de embedding a usar (default: "gemini-embedding-001")
            dimension: Dimensión del embedding (default: 2048, máximo 2048)
            text_key: Clave del diccionario que contiene el texto a embedear (default: "text")
            
        Returns:
            Lista de documentos con el campo 'embedding' agregado
            
        Raises:
            ValueError: Si algún documento no tiene la clave de texto o si la dimensión excede el máximo
        """
        try:
            # Validar que todos los documentos tengan la clave de texto
            for i, doc in enumerate(documents):
                if text_key not in doc:
                    error_msg = f"El documento en índice {i} no tiene la clave '{text_key}' requerida"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            # Generar embeddings
            texts = [doc[text_key] for doc in documents]
            embeddings = self.embed_texts(
                texts=texts,
                embedding_model=embedding_model,
                dimension=dimension
            )
            
            # Agregar embeddings a los documentos (campo fijo "embedding")
            for i, doc in enumerate(documents):
                doc[self.EMBEDDING_KEY] = embeddings[i]
                logger.debug(f"Embedding agregado al documento {i}")
            
            logger.info(f"Embeddings agregados a {len(documents)} documentos")
            return documents
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error al embedear documentos: {e}")
            raise

    def add_documents(self, documents: list[dict], embedding_model: str = "gemini-embedding-001", dimension: int = 2048, text_key: str = "text"):
        """
        Agrega documentos con embeddings a la colección de Firestore.
        
        Args:
            documents: Lista de diccionarios con los documentos
            embedding_model: Modelo de embedding a usar (default: "gemini-embedding-001")
            dimension: Dimensión del embedding (default: 2048, máximo 2048)
            text_key: Clave del diccionario que contiene el texto a embedear (default: "text")
            
        Raises:
            ValueError: Si algún documento no tiene el campo 'embedding' o si hay errores de validación
            GoogleCloudError: Si hay errores al escribir en Firestore
        """
        try:
            # Generar embeddings si no existen
            documents_with_embeddings = self.embed_documents(
                documents=documents,
                embedding_model=embedding_model,
                dimension=dimension,
                text_key=text_key
            )
            
            # Verificar que todos tengan el campo embedding
            for i, doc in enumerate(documents_with_embeddings):
                if self.EMBEDDING_KEY not in doc:
                    error_msg = f"El documento en índice {i} no tiene el campo '{self.EMBEDDING_KEY}'"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            # Agregar documentos a la colección usando batch (máximo 500 por batch)
            BATCH_SIZE = 500
            total_docs = len(documents_with_embeddings)
            added_count = 0
            
            # Dividir en lotes de 500 documentos
            for i in range(0, total_docs, BATCH_SIZE):
                batch = self.db.batch()
                batch_docs = documents_with_embeddings[i:i + BATCH_SIZE]
                
                for doc in batch_docs:
                    # Agregar metadatos obligatorios
                    doc_id = self._generate_document_id(doc, text_key)
                    doc['id'] = doc_id
                    doc['created_at'] = datetime.utcnow().isoformat()
                    
                    # Usar el hash como document ID en Firestore
                    doc_ref = self.db.collection(self.collection).document(doc_id)
                    batch.set(doc_ref, doc)
                
                batch.commit()
                added_count += len(batch_docs)
                logger.info(f"Lote agregado: {len(batch_docs)} documentos. Total: {added_count}/{total_docs}")
            
            logger.info(f"{total_docs} documentos agregados exitosamente a la colección '{self.collection}'")
            
        except ValueError:
            raise
        except GoogleCloudError as e:
            logger.error(f"Error de Firestore al agregar documentos: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado al agregar documentos: {e}")
            raise

    def get_documents(self):
        """
        Obtiene todos los documentos de la colección.
        
        Returns:
            Lista de diccionarios con los documentos
            
        Raises:
            GoogleCloudError: Si hay errores al consultar Firestore
        """
        try:
            docs = [doc.to_dict() for doc in self.db.collection(self.collection).stream()]
            logger.info(f"Obtenidos {len(docs)} documentos de la colección '{self.collection}'")
            return docs
        except GoogleCloudError as e:
            logger.error(f"Error de Firestore al obtener documentos: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado al obtener documentos: {e}")
            raise

    def get_document_by_key(self, key: str, value: Union[str, list, dict]):
        """
        Busca documentos por una llave y valor específicos.
        
        Args:
            key: Nombre del campo a buscar
            value: Valor a buscar
            
        Returns:
            Lista de diccionarios con los documentos encontrados
            
        Raises:
            GoogleCloudError: Si hay errores al consultar Firestore
        """
        try:
            docs = self.db.collection(self.collection).where(key, "==", value).get()
            result = [doc.to_dict() for doc in docs]
            logger.info(f"Encontrados {len(result)} documentos con {key}={value}")
            return result
        except GoogleCloudError as e:
            logger.error(f"Error de Firestore al buscar documentos por llave: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado al buscar documentos: {e}")
            raise
    
    def delete_document_by_key(self, key: str, value: Union[str, list, dict], batch_size: int = 500):
        """
        Elimina documentos por llave y valor específicos.
        
        ADVERTENCIA: Esta operación es irreversible.
        
        Args:
            key: Nombre del campo a buscar
            value: Valor a buscar
            batch_size: Número de documentos a eliminar por lote (máximo 500)
            
        Returns:
            Número de documentos eliminados
            
        Raises:
            ValueError: Si batch_size excede 500 o no se encuentran documentos
            GoogleCloudError: Si hay errores al eliminar en Firestore
        """
        try:
            if batch_size > 500:
                error_msg = "batch_size no puede exceder 500 (límite de Firestore)"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Buscar documentos que coincidan
            doc_refs = self.db.collection(self.collection).where(key, "==", value).stream()
            doc_refs_list = list(doc_refs)
            
            if not doc_refs_list:
                logger.warning(f"No se encontraron documentos con {key}={value}")
                return 0
            
            logger.warning(f"Eliminando {len(doc_refs_list)} documentos con {key}={value}")
            
            # Eliminar en batches
            deleted_count = 0
            total_docs = len(doc_refs_list)
            
            for i in range(0, total_docs, batch_size):
                batch = self.db.batch()
                batch_refs = doc_refs_list[i:i + batch_size]
                
                for doc_ref in batch_refs:
                    batch.delete(doc_ref.reference)
                
                batch.commit()
                deleted_count += len(batch_refs)
                logger.info(f"Eliminados {len(batch_refs)} documentos. Total: {deleted_count}/{total_docs}")
            
            logger.info(f"Eliminación completada: {deleted_count} documentos con {key}={value}")
            return deleted_count
            
        except ValueError:
            raise
        except GoogleCloudError as e:
            logger.error(f"Error de Firestore al eliminar documentos por llave: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado al eliminar documentos: {e}")
            raise
    
    def delete_collection(self, batch_size: int = 500):
        """
        Elimina todos los documentos de la colección.
        
        ADVERTENCIA: Esta operación es irreversible y eliminará todos los documentos.
        
        Args:
            batch_size: Número de documentos a eliminar por lote (máximo 500)
            
        Returns:
            Número total de documentos eliminados
            
        Raises:
            ValueError: Si batch_size excede 500
            GoogleCloudError: Si hay errores al eliminar en Firestore
        """
        try:
            if batch_size > 500:
                error_msg = "batch_size no puede exceder 500 (límite de Firestore)"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.warning(f"Iniciando eliminación de todos los documentos en la colección '{self.collection}'")
            
            deleted_count = 0
            collection_ref = self.db.collection(self.collection)
            
            while True:
                # Obtener un lote de documentos
                docs = list(collection_ref.limit(batch_size).stream())
                
                if not docs:
                    break  # No hay más documentos
                
                # Eliminar el lote usando batch
                batch = self.db.batch()
                for doc in docs:
                    batch.delete(doc.reference)
                
                batch.commit()
                deleted_count += len(docs)
                logger.info(f"Eliminados {len(docs)} documentos. Total: {deleted_count}")
            
            logger.info(f"Colección '{self.collection}' limpiada exitosamente. Total eliminados: {deleted_count}")
            return deleted_count
            
        except ValueError:
            raise
        except GoogleCloudError as e:
            logger.error(f"Error de Firestore al eliminar colección: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado al eliminar colección: {e}")
            raise

    def as_retriever(self, query:str, k:int = 5, model:str = "gemini-embedding-001", dimension:int = 2048, distance_measure:DistanceMeasure = DistanceMeasure.COSINE, filters:dict = None):
        """
        Búsqueda de similaridad vectorial optimizada con filtros opcionales.
        
        Args:
            query: Texto de búsqueda
            k: Número de resultados a retornar
            model: Modelo de embeddings a usar
            dimension: Dimensión de los embeddings
            distance_measure: Medida de distancia (COSINE, EUCLIDEAN, DOT_PRODUCT)
            filters: Diccionario de filtros {campo: valor} para aplicar condiciones de igualdad
                     Ejemplo: {"categoria": "cardiologia", "activo": True}
        
        Returns:
            Lista de documentos similares con sus scores
        """
        
        # Generar embedding del query
        vector_list = self.embed_texts(texts=[query], embedding_model=model, dimension=dimension)
        
        # Iniciar la consulta en la colección
        collection_ref = self.db.collection(self.collection)
        
        # Aplicar filtros si se proporcionan
        if filters:
            for field, value in filters.items():
                collection_ref = collection_ref.where(field, "==", value)
                logger.info(f"Filtro aplicado: {field} == {value}")
        
        # Búsqueda vectorial con filtros aplicados
        response = collection_ref.find_nearest(
            vector_field=self.EMBEDDING_KEY,
            query_vector=vector_list[0],
            distance_measure=distance_measure,
            limit=k
        )
        
        results = response.get()
        
        # Procesar resultados con list comprehension (más rápido)
        result_list = [
            {
                **{k: v for k, v in doc.to_dict().items() 
                   if k not in [self.EMBEDDING_KEY, "vector_distance"]},
                "score": 1 - doc.to_dict().get("vector_distance", 0)
            }
            for doc in results
        ]
        
        logger.info(f"Búsqueda completada: {len(result_list)} resultados encontrados")
        return result_list

# Clase de vector store
vector_store = FirestoreVectorStore(
    project="ace-line-451722-u7",
    database="(default)",
    collection="vector-store"
)
