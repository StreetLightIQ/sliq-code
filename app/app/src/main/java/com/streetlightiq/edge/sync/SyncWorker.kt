package com.streetlightiq.edge.sync

import android.content.Context
import androidx.work.*
import com.streetlightiq.edge.app.EdgeApplication
import com.streetlightiq.edge.data.local.RecordRepository
import com.streetlightiq.edge.data.remote.*
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.concurrent.TimeUnit
import java.io.File
import com.streetlightiq.edge.domain.wireJson
import kotlinx.serialization.json.*

class SyncEngine(private val records:RecordRepository,private val cloud:SupabaseCloud) {
    private val mutex=Mutex()
    @Volatile var lastMessage="Sign in to connect the backend."
        private set
    suspend fun sync():Boolean = mutex.withLock {
        val user=cloud.vault.read() ?: run { lastMessage="Sign in in Settings to upload."; return@withLock false }
        // Drain bounded batches. A follow-up worker drains longer offline backlogs.
        repeat(8) {
            val pending=records.dao.pending(user.userId)
            if(pending.isEmpty()) { lastMessage="Upload queue drained; check the failed-record count."; return@withLock true }
            val batch=pending.filter { record ->
                val extras=record.evidenceExtras?.let { wireJson.parseToJsonElement(it).jsonArray.map { ref -> ref.jsonObject.getValue("local").jsonPrimitive.content } } ?: emptyList()
                val available=(record.evidenceLocalPath==null || File(record.evidenceLocalPath).isFile) && extras.all { File(it).isFile }
                if(!available) records.dao.fail(record.id,record.revision,"PERMANENT_FAILURE","Evidence file missing; this record needs attention.")
                available
            }
            if(batch.isEmpty()) return@repeat
            try {
                val ack=try { cloud.upload(batch) } catch(e:CloudFailure) {
                    // A request-level 400 rejects the whole batch. Send records one by one so a single bad record cannot block the rest.
                    if(e.status!=400 || batch.size==1) throw e
                    val accepted=mutableListOf<Acknowledgement>(); val rejected=mutableListOf<Rejection>()
                    for(r in batch) try { cloud.upload(listOf(r)).let { accepted+=it.accepted; rejected+=it.rejected } }
                        catch(single:CloudFailure) { if(single.status==400) rejected+=Rejection(r.id,r.revision,single.message) else throw single }
                    BatchAck(accepted,rejected)
                }
                val accepted=ack.accepted.map { it.id to it.revision }.toSet()
                val rejected=ack.rejected.associateBy { it.id to it.revision }
                for(r in batch) when {
                    (r.id to r.revision) in accepted -> records.dao.acknowledge(r.id,r.revision)
                    (r.id to r.revision) in rejected -> records.dao.fail(r.id,r.revision,"PERMANENT_FAILURE",rejected.getValue(r.id to r.revision).reason)
                    else -> records.dao.fail(r.id,r.revision,"RETRYABLE_FAILURE","Backend did not acknowledge this revision.")
                }
                lastMessage="Uploaded ${accepted.size} records; ${rejected.size} rejected."
            } catch(e:Exception) {
                val permanent=e is CloudFailure && e.status in setOf(400,403,404,422)
                val hint=e.message ?: "Network unavailable; records remain saved on this phone."
                for(r in batch) records.dao.fail(r.id,r.revision,if(permanent) "PERMANENT_FAILURE" else "RETRYABLE_FAILURE",hint)
                lastMessage=hint
                return@withLock permanent
            }
        }
        false
    }
}
class ObservationSyncWorker(context:Context,parameters:WorkerParameters):CoroutineWorker(context,parameters) {
    override suspend fun doWork():Result {
        val engine=(applicationContext as EdgeApplication).syncEngine
        return if(engine.sync()) Result.success() else Result.retry()
    }
}
class SyncScheduler(private val context:Context) {
    private val constraints=Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build()
    fun immediate() {
        val request=OneTimeWorkRequestBuilder<ObservationSyncWorker>().setConstraints(constraints)
            .setBackoffCriteria(BackoffPolicy.EXPONENTIAL,10,TimeUnit.SECONDS).build()
        WorkManager.getInstance(context).enqueueUniqueWork("edge-live-sync",ExistingWorkPolicy.KEEP,request)
    }
    fun periodic() {
        WorkManager.getInstance(context).enqueueUniquePeriodicWork("edge-resume-sync",ExistingPeriodicWorkPolicy.KEEP,
            PeriodicWorkRequestBuilder<ObservationSyncWorker>(15,TimeUnit.MINUTES).setConstraints(constraints).build())
    }
}
