from typing import Any, AsyncIterator, Dict, Iterator, Optional, Sequence, Tuple
from google.cloud import firestore
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, CheckpointTuple, ChannelVersions
import pickle
from datetime import datetime
import pytz
import logging
import time

# Configure logging
logger = logging.getLogger(__name__)

class JsonPlusSerializerCompat(JsonPlusSerializer):
    """
    Serializador personalizado con compatibilidad hacia atrás para pickle.
    """
    def dumps(self, obj: Any) -> bytes:
        """
        Serializa un objeto a bytes usando dumps_typed y retorna solo los bytes.
        """
        _, data = self.dumps_typed(obj)
        return data
    
    def loads(self, data: bytes) -> Any:
        # Check if data appears to be pickle (backwards compatibility)
        if data.startswith(b"\x80") and data.endswith(b"."):
            logger.warning("Deserializando datos con pickle (compatibilidad hacia atrás).")
            try:
                return pickle.loads(data)
            except pickle.UnpicklingError as e:
                logger.error(f"Error al deserializar con pickle: {e}")
                raise ValueError("Error al deserializar datos antiguos (pickle).") from e
        # Try to deserialize using loads_typed with msgpack (default format)
        # Since dumps() uses dumps_typed which returns msgpack by default
        try:
            return self.loads_typed(("msgpack", data))
        except Exception as e:
            # Fallback: try json format (for very old data)
            try:
                return self.loads_typed(("json", data))
            except Exception:
                logger.error(f"Error al deserializar datos: {e}")
                raise
    
class FirestoreSaver(BaseCheckpointSaver):
    """
    Clase para implementar memoria de Langgraph en Firestore, debe especificarse 
    la base de datos (database) el nombre de coleccion de los checkpoints (collection_name)
    y el nombre de la coleccion de pasos intermedios (pw_collection_name).
    """
    serde = JsonPlusSerializerCompat()

    def __init__(self, database = "(default)", collection_name: str = "checkpoints", pw_collection_name: str = "checkpoint_writes", serde: Optional[Any] = None) -> None:
        super().__init__(serde=serde)
        self.database = database
        self.db: firestore.Client = firestore.Client(database = database)
        self.async_db: firestore.AsyncClient = firestore.AsyncClient(database = database)
        self.collection_name: str = collection_name
        self.pw_collection_name: str = pw_collection_name
        
        logger.info(f"FirestoreSaver inicializado. Colección Checkpoints: '{collection_name}', Colección Writes: '{pw_collection_name}'")

    # Método para traer memoria asociada a un thread_id
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        try:
            thread_id: str = config["configurable"]["thread_id"]
            thread_ts: Optional[str] = config["configurable"].get("thread_ts")
            
            doc_ref: firestore.DocumentReference = self.db.collection(self.collection_name).document(thread_id)
            doc: firestore.DocumentSnapshot = doc_ref.get()
            #Trae todo lo asociado al thead_id

            if not doc.exists:
                return None

            data: Dict[str, Any] = doc.to_dict()
            return self._process_checkpoint_data_common(data)
        except Exception as e:
            logger.error(f"Error retrieving checkpoint: {e}")
            return None

    # Método asincronico para traer memmoria
    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        try:
            thread_id: str = config["configurable"]["thread_id"]
            thread_ts: Optional[str] = config["configurable"].get("thread_ts")
            
            doc_ref: firestore.AsyncDocumentReference = self.async_db.collection(self.collection_name).document(thread_id)
            doc: firestore.DocumentSnapshot = await doc_ref.get()
            
            if not doc.exists:
                return None
                
            data: Dict[str, Any] = doc.to_dict()
            return self._process_checkpoint_data_common(data)
        except Exception as e:
            logger.error(f"Error retrieving checkpoint async: {e}")
            return None

    # Para listar checkpoints (Para listar checkpoints basados en un criterio)
    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        try:
            thread_id: Optional[str] = config["configurable"]["thread_id"] if config else None
            if filter:
                raise NotImplementedError("No se cuenta con la funcionalidad de filtrado")
            
            # Obtiene una referencia a la colección de checkpoints
            col_ref: firestore.CollectionReference = self.db.collection(self.collection_name)
            
            # Si se proporcionó un thread_id, filtra por ese thread_id
            if thread_id:
                col_ref = col_ref.where("thread_id", "==", thread_id)
            
            docs: firestore.QuerySnapshot = col_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit or 100).get()
            
            for doc in docs:
                try:
                    yield self._process_checkpoint_data_common(doc.to_dict())
                except Exception as e:
                    logger.error(f"Error deserializing checkpoint {doc.id}: {e}")
                    continue
        except Exception as e:
            logger.error(f"Error listing checkpoints: {e}")

    # Método asincronico  para listar checkpoints (Para listar checkpoints basados en un criterio)
    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        try:
            thread_id: Optional[str] = config["configurable"]["thread_id"] if config else None
            if filter:
                raise NotImplementedError("Filtering is not implemented for FirestoreSaver")
            
            # Obtiene una referencia a la colección de checkpoints
            col_ref: firestore.AsyncCollectionReference = self.async_db.collection(self.collection_name)
            
            # Si se proporcionó un thread_id, filtra por ese thread_id
            if thread_id:
                col_ref = col_ref.where("thread_id", "==", thread_id)
            
            docs: firestore.QuerySnapshot = await col_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit or 100).get()
            
            async for doc in docs:
                try:
                    yield self._process_checkpoint_data_common(doc.to_dict())
                except Exception as e:
                    logger.error(f"Error deserializing checkpoint {doc.id}: {e}")
                    continue
        except Exception as e:
            logger.error(f"Error listing checkpoints async: {e}")

    # Para guardar un checkpoint
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        try:
            thread_id: str = config["configurable"]["thread_id"]
            timestamp: str = datetime.now(pytz.timezone('America/Bogota')).strftime('%Y-%m-%d %H:%M:%S')
            # Mantener compatibilidad: usar "id" si existe, sino "ts"
            ts: str = checkpoint.get("id") or checkpoint.get("ts", "")
            
            doc_ref: firestore.DocumentReference = self.db.collection(self.collection_name).document(thread_id)
            doc_ref.set({
                "checkpoint": self.serde.dumps(checkpoint),
                "metadata": self.serde.dumps(metadata),
                "thread_id": thread_id,
                "timestamp": timestamp
            })
            
            logger.info(f"Checkpoint guardado exitosamente para thread_id: {thread_id}")

            return {
                "configurable": {
                    "thread_id": thread_id,
                    "thread_ts": ts,
                },
            }
        except Exception as e:
            logger.error(f"Error al guardar checkpoint para thread_id {thread_id}: {e}")
            raise IOError(f"No se pudo guardar el checkpoint para {thread_id}") from e

    # Método asincronico para put
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        try:
            thread_id: str = config["configurable"]["thread_id"]
            timestamp: str = datetime.now(pytz.timezone('America/Bogota')).strftime('%Y-%m-%d %H:%M:%S')
            # Mantener compatibilidad: usar "id" si existe, sino "ts"
            ts: str = checkpoint.get("id") or checkpoint.get("ts", "")
            
            doc_ref: firestore.AsyncDocumentReference = self.async_db.collection(self.collection_name).document(thread_id)
            await doc_ref.set({
                "checkpoint": self.serde.dumps(checkpoint),
                "metadata": self.serde.dumps(metadata),
                "thread_id": thread_id,
                "timestamp": timestamp
            })
            
            logger.info(f"Checkpoint guardado exitosamente (async) para thread_id: {thread_id}")

            return {
                "configurable": {
                    "thread_id": thread_id,
                    "thread_ts": ts,
                },
            }
        except Exception as e:
            logger.error(f"Error al guardar checkpoint async para thread_id {thread_id}: {e}")
            raise IOError(f"No se pudo guardar el checkpoint para {thread_id}") from e
    
    def put_writes(
        self,
        config: dict,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """
        Guarda escrituras intermedias vinculados a un checkpoint.


        Args:
            config (dict): Configuración del checkpoint.
            writes (Sequence[Tuple[str, Any]]): Lista de escrituras intermedias, cada uno como una pareja (channel, value).
            task_id (str): Identificador de la tarea creando las escrituras intermedias.
        """
        try:
            thread_id = config["configurable"]["thread_id"]
            thread_ts = config["configurable"].get("thread_ts")
            
            # Optimización: guardar todos los writes en un solo documento
            doc_id = f"{thread_id}_{task_id}"
            doc_ref = self.db.collection(self.pw_collection_name).document(doc_id)
            
            writes_data = {
                "thread_id": thread_id,
                "thread_ts": thread_ts,
                "task_id": task_id,
                "writes": [{"channel": channel, "value": self.serde.dumps(value)} for channel, value in writes],
                "timestamp": datetime.now(pytz.timezone("America/Bogota")).strftime('%Y-%m-%d %H:%M:%S')
            }
            
            doc_ref.set(writes_data)
            logger.debug(f"Pending writes stored for {thread_id}_{task_id}")
            
        except Exception as e:
            logger.error(f"Error storing pending writes: {e}")
            raise

    async def aput_writes(self, config: dict, writes: Sequence[Tuple[str, Any]], task_id: str) -> None:
        """Async version of put_writes."""
        return self.put_writes(config, writes, task_id)

    def list_writes(self, config: dict, *, before: Optional[dict] = None, limit: Optional[int] = None) -> Iterator[Tuple[str, str, Any]]:
        """List pending writes from Firestore."""
        try:
            thread_id = config["configurable"]["thread_id"]
            
            collection_ref = self.db.collection(self.pw_collection_name)
            collection_ref = collection_ref.where("thread_id", "==", thread_id)
            
            if before:
                before_ts = before["configurable"].get("thread_ts")
                if before_ts:
                    collection_ref = collection_ref.where("timestamp", "<", before_ts)
            
            collection_ref = collection_ref.order_by("timestamp", direction=firestore.Query.DESCENDING)
            
            if limit:
                collection_ref = collection_ref.limit(limit)
            
            docs = collection_ref.stream()
            
            for doc in docs:
                data = doc.to_dict()
                task_id = data["task_id"]
                for write_data in data["writes"]:
                    channel = write_data["channel"]
                    value = self.serde.loads(write_data["value"])
                    yield (task_id, channel, value)
                    
        except Exception as e:
            logger.error(f"Error listing pending writes: {e}")

    async def alist_writes(self, config: dict, *, before: Optional[dict] = None, limit: Optional[int] = None) -> AsyncIterator[Tuple[str, str, Any]]:
        """Async version of list_writes."""
        for item in self.list_writes(config, before=before, limit=limit):
            yield item

    def _process_checkpoint_data_common(self, data: Dict[str, Any]) -> CheckpointTuple:
        checkpoint: Checkpoint = self.serde.loads(data["checkpoint"])
        metadata: CheckpointMetadata = self.serde.loads(data["metadata"])
        thread_id: str = data["thread_id"]
        thread_ts: str = data["timestamp"]

        config: RunnableConfig = {"configurable": {"thread_id": thread_id, "thread_ts": thread_ts}}
        return CheckpointTuple(config=config, checkpoint=checkpoint, metadata=metadata, parent_config=None)


class OptimizedFirestoreSaver(FirestoreSaver):
    """
    FirestoreSaver optimizado que reduce la frecuencia de checkpoints para mejorar rendimiento.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_strategy = "minimal"  # Menos checkpoints para mejor performance
        self.last_checkpoint_time = {}  # Track timing per thread
        self.min_checkpoint_interval = 5.0  # Minimum 5 seconds between checkpoints (más eficiente)
        self.critical_nodes = {
            # Nodos de entrada principales
            "inicio_grafo_madre",           # Punto de entrada del sistema
            "get_token",                    # Generación de token de autenticación
            
            # Agentes principales (checkpoint después de cada uno)
            "agente_consulta",              # Consulta inicial de proveedor y póliza
            "agente_consulta_servicios",    # Selección de servicios
            "agente_consulta_diagnosticos", # Selección de diagnósticos
            "agente_autorizaciones",        # Cálculo y autorización final
            
            # Nodos de redirección importantes
            "dummie_redireccion",          # Punto de decisión entre agentes
            
            # Nodos de finalización
            "finalizar_autorizacion",      # Finalización del proceso
            "finalizar_seleccion_diagnosticos", # Finalización de selección de diagnósticos
        }
        
    def should_checkpoint(self, config: RunnableConfig, checkpoint: Checkpoint) -> bool:
        """
        Decide if we should save this checkpoint based on optimization strategy.
        """
        thread_id = config.get("configurable", {}).get("thread_id", "unknown")
        current_time = time.time()
        
        # Strategy 1: Check timing interval
        last_time = self.last_checkpoint_time.get(thread_id, 0)
        time_since_last = current_time - last_time
        
        # Strategy 2: Check if this is a critical node
        checkpoint_data = str(checkpoint.get("channel_values", {}))
        is_critical = any(node in checkpoint_data for node in self.critical_nodes)
        
        # Strategy 3: Always checkpoint at conversation boundaries
        is_conversation_boundary = "final_response" in checkpoint_data
        
        # Decision logic
        if self.checkpoint_strategy == "minimal":
            should_save = is_critical or is_conversation_boundary
        elif self.checkpoint_strategy == "selective":
            should_save = is_critical or is_conversation_boundary or time_since_last >= self.min_checkpoint_interval
        else:  # "all"
            should_save = True
        
        if should_save:
            self.last_checkpoint_time[thread_id] = current_time
        else:
            logger.debug(f"Checkpoint skipped for optimization: {thread_id}")
            
        return should_save
    
    def put(self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata, 
            new_versions: ChannelVersions) -> RunnableConfig:
        """Override put to add checkpoint optimization logic."""
        if not self.should_checkpoint(config, checkpoint):
            # Skip saving but return correct config format that LangGraph expects
            thread_id = config["configurable"]["thread_id"]
            ts = checkpoint.get("id") or checkpoint.get("ts", "")
            return {
                "configurable": {
                    "thread_id": thread_id,
                    "thread_ts": ts,
                },
            }
        
        # If we should checkpoint, use the parent implementation
        return super().put(config, checkpoint, metadata, new_versions)
    
    async def aput(self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata,
                   new_versions: ChannelVersions) -> RunnableConfig:
        """Override async put with same optimization logic."""
        if not self.should_checkpoint(config, checkpoint):
            # Skip saving but return correct config format that LangGraph expects
            thread_id = config["configurable"]["thread_id"]
            ts = checkpoint.get("id") or checkpoint.get("ts", "")
            return {
                "configurable": {
                    "thread_id": thread_id,
                    "thread_ts": ts,
                },
            }
        
        return await super().aput(config, checkpoint, metadata, new_versions)