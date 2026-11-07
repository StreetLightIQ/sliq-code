package com.streetlightiq.edge.sensors

import android.content.*
import android.Manifest
import android.content.pm.PackageManager
import androidx.core.content.ContextCompat
import android.hardware.*
import android.location.*
import android.os.*
import com.streetlightiq.edge.domain.*
import kotlinx.serialization.json.*
import java.io.*
import java.security.MessageDigest
import java.util.zip.GZIPOutputStream

/** Independent of inference: monotonic sensor times, bounded sampling, streamed rotating files. */
class SurveyRecorder(private val context:Context):SensorEventListener {
    private val manager=context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val thread=HandlerThread("survey-sensors").apply { start() }
    private val handler=Handler(thread.looper)
    private var writer:BufferedWriter?=null
    private var directory:File?=null
    private var part=0
    private var size=0L
    private var originNs=0L
    private var originUtc=0L
    private val last=mutableMapOf<Sensor,Long>()
    private val latest=linkedMapOf<String,JsonObject>()
    private var failure:String?=null
    private var active=false
    private val lm=context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
    private val status=object:GnssStatus.Callback() {
        override fun onSatelliteStatusChanged(s:GnssStatus) { event("gnss_status",obj("satellites" to JsonArray((0 until s.satelliteCount).map { i ->
            obj("svid" to j(s.getSvid(i)),"constellation" to j(s.getConstellationType(i)),"cn0_dbhz" to j(s.getCn0DbHz(i)),
                "azimuth_deg" to j(s.getAzimuthDegrees(i)),"elevation_deg" to j(s.getElevationDegrees(i)),"used_in_fix" to j(s.usedInFix(i))) }))) }
    }
    private val measurements=object:GnssMeasurementsEvent.Callback() {
        override fun onGnssMeasurementsReceived(e:GnssMeasurementsEvent) { event("gnss_measurements",obj(
            "clock_time_ns" to j(e.clock.timeNanos),"full_bias_ns" to j(if(e.clock.hasFullBiasNanos()) e.clock.fullBiasNanos else null),
            "bias_ns" to j(if(e.clock.hasBiasNanos()) e.clock.biasNanos else null),
            "hardware_discontinuity_count" to j(e.clock.hardwareClockDiscontinuityCount),
            "measurements" to JsonArray(e.measurements.map { m -> obj("svid" to j(m.svid),"constellation" to j(m.constellationType),
                "state" to j(m.state),"time_offset_ns" to j(m.timeOffsetNanos),"received_sv_time_ns" to j(m.receivedSvTimeNanos),
                "received_sv_time_uncertainty_ns" to j(m.receivedSvTimeUncertaintyNanos),"cn0_dbhz" to j(m.cn0DbHz),
                "pseudorange_rate_mps" to j(m.pseudorangeRateMetersPerSecond),"pseudorange_rate_uncertainty_mps" to j(m.pseudorangeRateUncertaintyMetersPerSecond),
                "accumulated_delta_range_m" to j(m.accumulatedDeltaRangeMeters),"accumulated_delta_range_state" to j(m.accumulatedDeltaRangeState),
                "carrier_frequency_hz" to j(if(m.hasCarrierFrequencyHz()) m.carrierFrequencyHz else null),"multipath_indicator" to j(m.multipathIndicator)) }))) }
    }
    private val deviceTick=object:Runnable {
        override fun run() {
            synchronized(this@SurveyRecorder) { if(!active) return }
            val battery=context.registerReceiver(null,IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            val power=context.getSystemService(Context.POWER_SERVICE) as PowerManager
            event("device_status",obj("battery_level" to j(battery?.getIntExtra(BatteryManager.EXTRA_LEVEL,-1)),
                "battery_scale" to j(battery?.getIntExtra(BatteryManager.EXTRA_SCALE,-1)),
                "battery_temperature_tenths_c" to j(battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE,-1)),
                "battery_voltage_mv" to j(battery?.getIntExtra(BatteryManager.EXTRA_VOLTAGE,-1)),
                "battery_status" to j(battery?.getIntExtra(BatteryManager.EXTRA_STATUS,-1)),
                "power_save" to j(power.isPowerSaveMode),"thermal_status" to j(if(Build.VERSION.SDK_INT>=29) power.currentThermalStatus else null),
                "available_bytes" to j(StatFs(context.filesDir.path).availableBytes),"wall_clock_utc_ms" to j(System.currentTimeMillis())))
            synchronized(this@SurveyRecorder) { runCatching { writer?.flush() }.onFailure { failure=it.message } }
            handler.postDelayed(this,5000)
        }
    }
    @Synchronized fun start(sessionId:String,model:JsonObject,config:JsonObject) {
        check(!active)
        directory=File(context.filesDir,"surveys/$sessionId").apply { mkdirs() }
        originNs=SystemClock.elapsedRealtimeNanos(); originUtc=System.currentTimeMillis()
        part=0; failure=null; last.clear(); latest.clear(); active=true; openPart()
        event("survey_header",obj("schema_version" to j(1),"session_id" to j(sessionId),"model" to model,"configuration" to config,
            "manufacturer" to j(Build.MANUFACTURER),"device_model" to j(Build.MODEL),"build_fingerprint" to j(Build.FINGERPRINT),
            "sdk" to j(Build.VERSION.SDK_INT),"clock_origin_monotonic_ns" to j(originNs),"clock_origin_utc_ms" to j(originUtc),
            "sensor_requested_period_us" to j(50000),"camera_capture_max_hz" to j(10),
            "ambient_lux_note" to j("Device light sensor; orientation/occlusion dependent, not road illuminance.")))
        val inventory=manager.getSensorList(Sensor.TYPE_ALL).map { s ->
            val registered=runCatching { manager.registerListener(this,s,50000,0,handler) }.getOrDefault(false)
            obj("name" to j(s.name),"vendor" to j(s.vendor),"type" to j(s.type),"string_type" to j(s.stringType),"version" to j(s.version),
                "resolution" to j(s.resolution),"maximum_range" to j(s.maximumRange),"power_ma" to j(s.power),
                "min_delay_us" to j(s.minDelay),"max_delay_us" to j(s.maxDelay),"reporting_mode" to j(s.reportingMode),
                "registered" to j(registered),"status" to j(if(registered) "registered" else "permission_or_reporting_mode_unsupported"))
        }
        event("sensor_inventory",obj("sensors" to JsonArray(inventory)))
        var gnss=false; var raw=false
        if(ContextCompat.checkSelfPermission(context,Manifest.permission.ACCESS_FINE_LOCATION)==PackageManager.PERMISSION_GRANTED) {
            gnss=runCatching { lm.registerGnssStatusCallback(status,handler) }.getOrDefault(false)
            raw=runCatching { lm.registerGnssMeasurementsCallback(measurements,handler) }.getOrDefault(false)
        }
        event("gnss_capabilities",obj("status_registered" to j(gnss),"raw_registered" to j(raw)))
        handler.post(deviceTick)
    }
    private fun openPart() {
        writer=File(directory,"telemetry-${(part++).toString().padStart(5,'0')}.jsonl.gz").outputStream().let { GZIPOutputStream(it) }.bufferedWriter()
        size=0
    }
    @Synchronized fun event(kind:String,data:JsonObject,ns:Long=SystemClock.elapsedRealtimeNanos()) {
        if(!active || failure!=null) return
        try {
            val line=obj("kind" to j(kind),"monotonic_ns" to j(ns),"utc_ms" to j(originUtc+(ns-originNs)/1000000),"data" to wireSafe(data)).toString()
            if(size>4L*1024*1024) { writer!!.close(); openPart() }
            writer!!.write(line); writer!!.newLine(); size+=line.length
        } catch(e:Exception) { failure=e.message ?: "telemetry_write_failed" }
    }
    override fun onSensorChanged(e:SensorEvent) {
        synchronized(this) {
            if(!active || e.timestamp-(last[e.sensor] ?: 0)<50000000L) return
            last[e.sensor]=e.timestamp
            val sample=obj("name" to j(e.sensor.name),"type" to j(e.sensor.type),"string_type" to j(e.sensor.stringType),
                "accuracy" to j(e.accuracy),"values" to JsonArray(e.values.map { if(it.isFinite()) j(it) else JsonNull }))
            latest[e.sensor.stringType+":"+e.sensor.name]=obj("monotonic_ns" to j(e.timestamp),"sample" to sample)
            event("sensor",sample,e.timestamp)
        }
    }
    override fun onAccuracyChanged(sensor:Sensor?,accuracy:Int) {}
    @Synchronized fun snapshot()=JsonObject(latest.toMap())
    @Synchronized fun finish(reason:String):List<File> {
        if(!active) return directory?.listFiles()?.sortedBy { it.name } ?: emptyList()
        event("survey_footer",obj("reason" to j(reason),"write_error" to j(failure)))
        active=false; manager.unregisterListener(this); handler.removeCallbacks(deviceTick)
        runCatching { lm.unregisterGnssStatusCallback(status); lm.unregisterGnssMeasurementsCallback(measurements) }
        runCatching { writer?.close() }.onFailure { failure=it.message }; writer=null
        val files=directory!!.listFiles()!!.filter { it.isFile && it.name!="manifest.json" }.sortedBy { it.name }
        File(directory,"manifest.json").writeText(obj("schema_version" to j(1),"complete" to j(failure==null),"write_error" to j(failure),
            "files" to JsonArray(files.map { file -> obj("name" to j(file.name),"bytes" to j(file.length()),"sha256" to j(sha256(file)),"mime_type" to j(when(file.extension) { "png" -> "image/png"; "json" -> "application/json"; else -> "application/gzip" })) })).toString())
        return directory!!.listFiles()!!.sortedBy { it.name }
    }
    companion object {
        /** Recover readable complete JSON lines from a gzip interrupted by process death. */
        fun recover(directory:File):List<File> {
            if(!directory.isDirectory) return emptyList()
            for(file in directory.listFiles().orEmpty().filter { it.extension=="gz" }) {
                val tmp=File(file.path+".recover")
                GZIPOutputStream(tmp.outputStream()).bufferedWriter().use { out ->
                    runCatching {
                        java.util.zip.GZIPInputStream(file.inputStream()).bufferedReader().use { input ->
                            while(true) {
                                val line=input.readLine() ?: break
                                if(runCatching { wireJson.parseToJsonElement(line) }.isSuccess) { out.write(line); out.newLine() }
                            }
                        }
                    }
                }
                check(tmp.renameTo(file))
            }
            val files=directory.listFiles().orEmpty().filter { it.isFile && it.name!="manifest.json" && !it.name.endsWith(".recover") }.sortedBy { it.name }
            File(directory,"manifest.json").writeText(obj("schema_version" to j(1),"complete" to j(false),"recovered" to j(true),
                "reason" to j("process_restarted; final buffered readings may be lost"),
                "files" to JsonArray(files.map { obj("name" to j(it.name),"bytes" to j(it.length()),"sha256" to j(sha256(it))) })).toString())
            return directory.listFiles().orEmpty().filter { it.isFile && !it.name.endsWith(".recover") }.sortedBy { it.name }
        }

        fun sha256(file:File):String {
            val digest=MessageDigest.getInstance("SHA-256")
            file.inputStream().use { input -> val buffer=ByteArray(65536); while(true) { val n=input.read(buffer); if(n<0) break; digest.update(buffer,0,n) } }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }
    }
}
