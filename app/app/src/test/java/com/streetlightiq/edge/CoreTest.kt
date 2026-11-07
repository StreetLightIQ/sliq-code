package com.streetlightiq.edge

import com.streetlightiq.edge.config.RuntimeConfig
import com.streetlightiq.edge.domain.*
import com.streetlightiq.edge.location.LocationBuffer
import com.streetlightiq.edge.evidence.PhysicsFramePicker
import com.streetlightiq.edge.tracking.*
import org.junit.Assert.*
import org.junit.Test

class CoreTest {
    private fun time(ms:Long)=FrameTime(ms,1700000000000+ms,ms*1_000_000,ms*1_000_000,"sensor_realtime",0,640,480)
    private fun detection(x:Float,y:Float=0.2f)=Detection(Box(x,y,x+0.03f,y+0.04f),0.8f,0,"lamp_head","yolo26_INT8")
    @Test fun globalAssignmentAvoidsGreedyMismatch() {
        val a=Hungarian.assign(arrayOf(doubleArrayOf(.3,.31,1.0,1.0),doubleArrayOf(.4,.9,1.0,1.0)))
        assertArrayEquals(intArrayOf(1,0),a)
    }
    @Test fun sameLampSurvivesMissingDetectionThenCompletesOnce() {
        val tracker=CentreScaleTracker(RuntimeConfig(maxMissingMs=1500))
        val first=tracker.update(listOf(detection(.2f)),time(1000)).active.single()
        assertEquals(first.id,tracker.update(listOf(detection(.22f)),time(1200)).active.single().id)
        assertTrue(tracker.update(emptyList(),time(1800)).completed.isEmpty())
        assertEquals(first.id,tracker.update(listOf(detection(.26f)),time(2000)).matched.single().id)
        assertEquals(first.id,tracker.update(emptyList(),time(3600)).completed.single().id)
        assertTrue(tracker.flush().isEmpty())
    }
    @Test fun distantObjectsAndDifferentSourcesAreNotMerged() {
        val tracker=CentreScaleTracker(RuntimeConfig())
        val first=tracker.update(listOf(detection(.1f)),time(1000)).active.single()
        val second=tracker.update(listOf(detection(.7f)),time(1200)).matched.single()
        assertNotEquals(first.observationId,second.observationId)
        assertEquals(2,tracker.flush().size)
        tracker.reset(); assertEquals(1L,tracker.update(listOf(detection(.1f)),time(1)).active.single().id)
    }
    @Test(expected=IllegalArgumentException::class) fun rejectsOutOfOrderFrames() {
        val tracker=CentreScaleTracker(RuntimeConfig())
        tracker.update(emptyList(),time(1000)); tracker.update(emptyList(),time(999))
    }
    @Test fun gpsUsesCaptureTimeAndRejectsStaleFixes() {
        val buffer=LocationBuffer(2)
        fun fix(ms:Long)=LocationSample(ms.toString(),18.5,73.8,5f,1700000000000+ms,ms*1_000_000,null,null,null,null,null,null,"fused",false)
        buffer.add(fix(1000)); buffer.add(fix(3000))
        assertEquals("3000",buffer.closest(2800_000_000,5000).sample?.id)
        assertEquals(200L,buffer.closest(2800_000_000,5000).offsetMs)
        assertNull(buffer.closest(10000_000_000,5000).sample)
        buffer.add(fix(5000)); assertEquals("3000",buffer.closest(1000_000_000,5000).sample?.id)
    }
    @Test fun normalizationAndWireRoundTripPreserveResearchFields() {
        val a=Box(.1f,.2f,.3f,.4f)
        assertEquals(1f,a.iou(a),1e-5f); assertEquals(0f,a.iou(Box(.7f,.7f,.8f,.8f)),1e-5f)
        val s=TrackSample("obs",3,time(3000),detection(.1f),null,null,"missing")
        assertEquals(s,wireJson.decodeFromString<TrackSample>(encoded(s).toString()))
        assertEquals(RuntimeConfig(),wireJson.decodeFromString<RuntimeConfig>(encoded(RuntimeConfig()).toString()))
    }
    @Test fun physicsFramesSpreadOverLampSizeAndStayCapped() {
        val picker=PhysicsFramePicker(maxFrames=5)
        var encodes=0
        // A lamp growing from 1% to 20% of the frame width over 60 frames, as the car approaches.
        for(i in 0 until 60) {
            val w=0.01f*Math.pow(20.0,i/59.0).toFloat()
            picker.observe(7,time(i*100L),Box(0.5f,0.1f,0.5f+w,0.1f+w),0.8f,obj()) { encodes++; byteArrayOf(i.toByte()) }
        }
        val kept=picker.frames(7)
        assertEquals(5,kept.size)
        assertEquals(0L,kept.first().frame.frame_id)                    // smallest (farthest) frame kept
        assertEquals(5900L,kept.last().frame.frame_id)                  // largest (nearest) frame kept
        assertEquals(5,kept.map { it.step }.distinct().size)
        val gaps=kept.zipWithNext().map { (a,b) -> b.sizeLog-a.sizeLog }
        assertTrue(gaps.max()<2.5*gaps.min())                             // roughly even spread
        assertTrue(encodes<30)                                          // JPEG only for frames that are kept
        // Same size, clearly better score: replaces; marginally better: ignored.
        picker.observe(8,time(1),Box(0f,0f,.1f,.1f),0.5f,obj()) { byteArrayOf(1) }
        picker.observe(8,time(2),Box(0f,0f,.1f,.1f),0.52f,obj()) { byteArrayOf(2) }
        assertEquals(1L,picker.frames(8).single().frame.frame_id)
        picker.observe(8,time(3),Box(0f,0f,.1f,.1f),0.7f,obj()) { byteArrayOf(3) }
        assertEquals(3L,picker.frames(8).single().frame.frame_id)
        picker.discard(8); assertTrue(picker.frames(8).isEmpty())
    }

    @Test fun uploadsAreAlwaysValidJson() {
        // The camera's Rational values used to be written unquoted ("109/64"), which made whole upload batches invalid.
        val bad=com.streetlightiq.edge.domain.wireJson.parseToJsonElement("""{"m":[109/64,-191/256,1/0],"n":NaN,"i":Infinity,"ok":1.5,"s":"1/2","b":true,"z":null}""")
        val safe=com.streetlightiq.edge.domain.wireSafe(bad).toString()
        org.junit.Assert.assertEquals("""{"m":[1.703125,-0.74609375,null],"n":null,"i":null,"ok":1.5,"s":"1/2","b":true,"z":null}""",safe)
        org.junit.Assert.assertEquals(kotlinx.serialization.json.JsonNull,com.streetlightiq.edge.domain.j(Double.NaN))
        org.junit.Assert.assertEquals(kotlinx.serialization.json.JsonNull,com.streetlightiq.edge.domain.j(Float.POSITIVE_INFINITY))
        val big=com.streetlightiq.edge.domain.obj("schema_version" to com.streetlightiq.edge.domain.j(1),"camera" to com.streetlightiq.edge.domain.obj(
            "all_characteristics" to com.streetlightiq.edge.domain.j("x".repeat(900_000)),"iso" to com.streetlightiq.edge.domain.j(3200)))
        val sent=com.streetlightiq.edge.data.remote.uploadPayload(big.toString())
        org.junit.Assert.assertTrue(sent.toString().length<com.streetlightiq.edge.data.remote.MAX_PAYLOAD_CHARS)
        org.junit.Assert.assertTrue(sent.toString().contains("\"iso\":3200") && sent.toString().contains("payload_slimmed"))
    }
}
