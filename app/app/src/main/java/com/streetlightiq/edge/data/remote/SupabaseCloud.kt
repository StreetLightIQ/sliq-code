package com.streetlightiq.edge.data.remote

import com.streetlightiq.edge.config.*
import com.streetlightiq.edge.data.local.OutboxEntity
import com.streetlightiq.edge.domain.*
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.json.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.RequestBody.Companion.asRequestBody
import retrofit2.Response
import retrofit2.Retrofit
import retrofit2.http.*
import java.io.File
import java.util.concurrent.TimeUnit

interface CloudEndpoints {
    @POST("auth/v1/token") suspend fun token(@Query("grant_type") grantType:String,
        @Header("apikey") key:String,@Body body:RequestBody):Response<ResponseBody>
    @POST("rest/v1/rpc/ingest_edge_records") suspend fun ingest(@Header("apikey") key:String,
        @Header("Authorization") auth:String,@Body body:RequestBody):Response<ResponseBody>
    @POST("storage/v1/object/evidence/{path}") suspend fun evidence(@Path(value="path",encoded=true) path:String,
        @Header("apikey") key:String,@Header("Authorization") auth:String,
        @Header("x-upsert") upsert:String="true",@Body body:RequestBody):Response<ResponseBody>
}
data class Acknowledgement(val id:String,val revision:Long)
data class Rejection(val id:String,val revision:Long,val reason:String)
data class BatchAck(val accepted:List<Acknowledgement>,val rejected:List<Rejection>)
class CloudFailure(val status:Int,override val message:String):Exception(message)
interface CloudRepository {
    suspend fun upload(records:List<OutboxEntity>):BatchAck
}
/** Server limit is 1 MB per record (pg_column_size). Stay well below it. */
const val MAX_PAYLOAD_CHARS=800_000
/** Bulky diagnostic camera dumps; the same data is in the session's telemetry archive. */
private val SLIM_KEYS=setOf("all_characteristics","all_result_keys")
private fun slim(e:JsonElement):JsonElement = when(e) {
    is JsonObject -> JsonObject(e.filterKeys { it !in SLIM_KEYS }.mapValues { slim(it.value) })
    is JsonArray -> JsonArray(e.map { slim(it) })
    else -> e
}
/** Stored payload -> what is sent: always valid JSON; slimmed when too large for the server. */
fun uploadPayload(stored:String):JsonElement {
    val safe=wireSafe(wireJson.parseToJsonElement(stored))
    if(safe.toString().length<=MAX_PAYLOAD_CHARS || safe !is JsonObject) return safe
    return JsonObject(slim(safe).jsonObject+mapOf("payload_slimmed" to j("camera dumps removed to fit the 1 MB record limit; see telemetry archive")))
}

class SupabaseCloud(private val preferences:Preferences, val vault:TokenVault):CloudRepository {
    private val refreshMutex=Mutex()
    private val http=OkHttpClient.Builder().connectTimeout(15,TimeUnit.SECONDS).readTimeout(30,TimeUnit.SECONDS).build()
    private fun api(config:BackendConfig):CloudEndpoints {
        require(config.url.startsWith("https://") && config.url.endsWith(".supabase.co")) { "Use your HTTPS Supabase Project URL" }
        require(config.publicKey.isNotBlank() && !config.publicKey.startsWith("sb_secret_")) { "Use a publishable or legacy anon key" }
        // Legacy service-role JWTs are also rejected rather than relying on their spelling.
        if(config.publicKey.count { it=='.' }==2) {
            val payload=config.publicKey.split('.')[1]
            val text=String(android.util.Base64.decode(payload,android.util.Base64.URL_SAFE or android.util.Base64.NO_PADDING or android.util.Base64.NO_WRAP))
            require((wireJson.parseToJsonElement(text) as JsonObject)["role"]?.jsonPrimitive?.content != "service_role")
        }
        return Retrofit.Builder().baseUrl(config.url.trimEnd('/')+"/").client(http).build().create(CloudEndpoints::class.java)
    }
    private fun body(value:JsonElement)=value.toString().toRequestBody("application/json".toMediaType())
    private fun read(response:Response<ResponseBody>):JsonObject {
        if(!response.isSuccessful) {
            val detail=runCatching { response.errorBody()?.use { it.string() } }.getOrNull()?.take(300)?.replace(Regex("\\s+")," ")
            val hint=when(response.code()) {
                400 -> "Bad request or invalid login. Check the email, password and backend schema."
                401 -> "Authentication expired. Sign in again."
                403 -> "Permission denied. Check backend row and storage policies."
                404 -> "Backend function or bucket missing. Run backend/setup.sql."
                429 -> "Backend rate limit; retry later."
                else -> "Backend HTTP ${response.code()}; retry or inspect project logs."
            }
            throw CloudFailure(response.code(),if(detail.isNullOrBlank()) hint else "$hint Server: $detail")
        }
        return wireJson.parseToJsonElement(response.body()!!.use { it.string() }) as JsonObject
    }
    private fun session(json:JsonObject,url:String):CloudSession {
        val user=json.getValue("user").jsonObject
        return CloudSession(user.getValue("id").jsonPrimitive.content,user["email"]?.jsonPrimitive?.content ?: "",
            url,json.getValue("access_token").jsonPrimitive.content,json.getValue("refresh_token").jsonPrimitive.content,
            System.currentTimeMillis()+json.getValue("expires_in").jsonPrimitive.long*1000)
    }
    suspend fun signIn(config:BackendConfig,email:String,password:String):CloudSession = refreshMutex.withLock {
        val json=read(api(config).token("password",config.publicKey,body(obj("email" to j(email.trim()),"password" to j(password)))))
        session(json,config.url).also { vault.save(it) }
    }
    private suspend fun validSession(config:BackendConfig):CloudSession = refreshMutex.withLock {
        val old=vault.read() ?: throw CloudFailure(401,"Sign in in Settings before uploading.")
        require(old.projectUrl==config.url) { "Sign in to the configured backend project." }
        if(old.expiresAtMs>System.currentTimeMillis()+60000) return@withLock old
        val json=read(api(config).token("refresh_token",config.publicKey,body(obj("refresh_token" to j(old.refreshToken)))))
        session(json,config.url).also { vault.save(it) }
    }
    override suspend fun upload(records:List<OutboxEntity>):BatchAck {
        val config=preferences.get().backend; val credentials=validSession(config); val endpoint=api(config)
        require(records.all { it.ownerId==credentials.userId }) { "Records belong to a different survey account." }
        for(r in records) {
            val files=mutableListOf<Pair<String,String>>()
            if(r.evidenceLocalPath!=null && r.evidenceCloudPath!=null) files+=r.evidenceLocalPath to r.evidenceCloudPath
            r.evidenceExtras?.let { text -> (wireJson.parseToJsonElement(text) as JsonArray).forEach {
                files+=it.jsonObject.getValue("local").jsonPrimitive.content to it.jsonObject.getValue("cloud").jsonPrimitive.content
            } }
            for((local,remote) in files) {
            val file=File(local)
            if(!file.isFile) throw CloudFailure(422,"Evidence file missing. Keep the app's data intact until it uploads.")
            val response=endpoint.evidence(remote,config.publicKey,"Bearer ${credentials.accessToken}",
                body=file.asRequestBody((when(file.extension.lowercase()) {
                    "png" -> "image/png"; "gz" -> "application/gzip"; "json" -> "application/json"; else -> "image/jpeg"
                }).toMediaType()))
            read(response)
            }
        }
        val entries=JsonArray(records.map { r -> obj("id" to j(r.id),"session_id" to j(r.sessionId),
            "kind" to j(r.kind),"revision" to j(r.revision),"payload" to uploadPayload(r.payload)) })
        val result=read(endpoint.ingest(config.publicKey,"Bearer ${credentials.accessToken}",body(obj("p_records" to entries))))
        return BatchAck(result.getValue("accepted").jsonArray.map { Acknowledgement(it.jsonObject.getValue("id").jsonPrimitive.content,it.jsonObject.getValue("revision").jsonPrimitive.long) },
            result.getValue("rejected").jsonArray.map { Rejection(it.jsonObject.getValue("id").jsonPrimitive.content,it.jsonObject.getValue("revision").jsonPrimitive.long,it.jsonObject.getValue("reason").jsonPrimitive.content) })
    }
}
