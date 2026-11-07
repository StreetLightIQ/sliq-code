package com.streetlightiq.edge.data.local

import androidx.room.withTransaction
import com.streetlightiq.edge.domain.*
import kotlinx.serialization.json.*
import com.streetlightiq.edge.sensors.SurveyRecorder
import java.io.File

class RecordRepository(private val db:EdgeDatabase) {
    val dao=db.dao()
    suspend fun session(entity:SessionEntity,extras:String?=null) = db.withTransaction {
        dao.putSession(entity)
        dao.enqueue(OutboxEntity(entity.id,entity.id,entity.ownerId,"SESSION",entity.revision,entity.payload,evidenceExtras=extras))
    }
    suspend fun observation(entity:ObservationEntity,ownerId:String,localFile:String?,cloudFile:String?,extras:String?=null) = db.withTransaction {
        dao.putObservation(entity)
        dao.enqueue(OutboxEntity(entity.id,entity.sessionId,ownerId,"OBSERVATION",entity.revision,entity.payload,localFile,cloudFile,evidenceExtras=extras))
    }
    suspend fun sampleRows(samples:List<TrackSampleEntity>) { if(samples.isNotEmpty()) dao.putSamples(samples) }
    suspend fun route(rows:List<LocationEntity>,record:OutboxEntity) = db.withTransaction { dao.putLocations(rows); dao.enqueue(record) }
    suspend fun recover(filesDir:File?=null) {
        for(s in dao.unfinishedSessions()) {
            val ended=System.currentTimeMillis()
            val files=filesDir?.let { SurveyRecorder.recover(File(it,"surveys/${s.id}")) } ?: emptyList()
            val extras=if(files.isEmpty()) null else JsonArray(files.map { file -> obj("local" to j(file.absolutePath),"cloud" to j("${s.ownerId}/${s.id}/survey/${file.name}")) }).toString()
            val archive=obj("recovered" to j(true),"complete" to j(false),"storage_bucket" to j("evidence"),
                "manifest_path" to j(if(files.isEmpty()) null else "${s.ownerId}/${s.id}/survey/manifest.json"),"total_files" to j(files.size))
            val payload=JsonObject(wireJson.parseToJsonElement(s.payload).let { it as JsonObject } +
                mapOf("survey_archive" to archive,"state" to j("INTERRUPTED"),"end_utc_ms" to j(ended),"interruption_reason" to j("process_restarted")))
            val obs=dao.observations(s.id)
            db.withTransaction {
                dao.putSession(s.copy(state="INTERRUPTED",endedMs=ended,revision=s.revision+1,payload=payload.toString()))
                dao.enqueue(OutboxEntity(s.id,s.id,s.ownerId,"SESSION",s.revision+1,payload.toString(),evidenceExtras=extras))
                for(o in obs.filter { !it.finished }) {
                    val p=JsonObject((wireJson.parseToJsonElement(o.payload) as JsonObject)+mapOf("finished" to j(true),"completion_reason" to j("process_restarted")))
                    val new=o.copy(finished=true,revision=o.revision+1,payload=p.toString())
                    dao.putObservation(new)
                    // The crop path remains in the last observation's evidence payload, so pending image upload is preserved.
                    val old=dao.pending(s.ownerId,10000).firstOrNull { it.id==o.id }
                    dao.enqueue(OutboxEntity(o.id,s.id,s.ownerId,"OBSERVATION",new.revision,new.payload,
                        old?.evidenceLocalPath,old?.evidenceCloudPath,evidenceExtras=old?.evidenceExtras))
                }
            }
        }
    }
}
