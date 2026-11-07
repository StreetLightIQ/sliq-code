package com.streetlightiq.edge.data.local

import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Entity(tableName="sessions")
data class SessionEntity(@PrimaryKey val id:String, val ownerId:String, val startedMs:Long,
    val endedMs:Long?, val state:String, val revision:Long, val payload:String)
@Entity(tableName="observations",indices=[Index("sessionId")])
data class ObservationEntity(@PrimaryKey val id:String,val sessionId:String,val trackId:Long,
    val finished:Boolean,val revision:Long,val payload:String)
@Entity(tableName="track_samples",indices=[Index("sessionId"),Index("observationId")])
data class TrackSampleEntity(@PrimaryKey(autoGenerate=true) val id:Long=0,val sessionId:String,
    val observationId:String,val monotonicNs:Long,val utcMs:Long,val payload:String)
@Entity(tableName="location_samples",indices=[Index("sessionId")])
data class LocationEntity(@PrimaryKey val id:String,val sessionId:String,val utcMs:Long,val payload:String)
@Entity(tableName="upload_queue",indices=[Index("ownerId"),Index("state")])
data class OutboxEntity(@PrimaryKey val id:String,val sessionId:String,val ownerId:String,val kind:String,
    val revision:Long,val payload:String,val evidenceLocalPath:String?=null,val evidenceCloudPath:String?=null,
    val createdMs:Long=System.currentTimeMillis(),val state:String="PENDING",val attempts:Int=0,val error:String?=null,
    val evidenceExtras:String?=null)

@Dao
interface EdgeDao {
    @Insert(onConflict=OnConflictStrategy.REPLACE) suspend fun putSession(session:SessionEntity)
    @Insert(onConflict=OnConflictStrategy.REPLACE) suspend fun putObservation(observation:ObservationEntity)
    @Insert suspend fun putSamples(samples:List<TrackSampleEntity>)
    @Insert(onConflict=OnConflictStrategy.IGNORE) suspend fun putLocations(locations:List<LocationEntity>)
    @Insert(onConflict=OnConflictStrategy.REPLACE) suspend fun enqueue(record:OutboxEntity)
    @Query("SELECT * FROM sessions ORDER BY startedMs DESC LIMIT 30") fun sessions():Flow<List<SessionEntity>>
    @Query("SELECT * FROM observations WHERE sessionId=:sessionId ORDER BY trackId") suspend fun observations(sessionId:String):List<ObservationEntity>
    @Query("SELECT * FROM sessions WHERE state IN ('RUNNING','PAUSED','INITIALIZING')") suspend fun unfinishedSessions():List<SessionEntity>
    @Query("SELECT * FROM upload_queue WHERE ownerId=:ownerId AND state IN ('PENDING','RETRYABLE_FAILURE') ORDER BY createdMs LIMIT :limit")
    suspend fun pending(ownerId:String,limit:Int=25):List<OutboxEntity>
    @Query("SELECT COUNT(*) FROM upload_queue WHERE state!='UPLOADED'") fun pendingCount():Flow<Int>
    @Query("SELECT COUNT(*) FROM upload_queue WHERE state='PERMANENT_FAILURE'") fun failedCount():Flow<Int>
    @Query("UPDATE upload_queue SET state='UPLOADED',error=NULL WHERE id=:id AND revision=:revision")
    suspend fun acknowledge(id:String,revision:Long):Int
    @Query("UPDATE upload_queue SET state=:state,attempts=attempts+1,error=:error WHERE id=:id AND revision=:revision")
    suspend fun fail(id:String,revision:Long,state:String,error:String)
    @Query("UPDATE upload_queue SET state='PENDING',error=NULL WHERE state IN ('RETRYABLE_FAILURE','PERMANENT_FAILURE')") suspend fun retryFailures()
    @Query("SELECT COUNT(*) FROM upload_queue WHERE state='UPLOADED'") suspend fun uploadedCount():Int
}

@Database(entities=[SessionEntity::class,ObservationEntity::class,TrackSampleEntity::class,LocationEntity::class,OutboxEntity::class],version=1,exportSchema=true)
abstract class EdgeDatabase:RoomDatabase() { abstract fun dao():EdgeDao }
