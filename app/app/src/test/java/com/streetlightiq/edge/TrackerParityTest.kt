package com.streetlightiq.edge

import com.streetlightiq.edge.config.RuntimeConfig
import com.streetlightiq.edge.domain.*
import com.streetlightiq.edge.tracking.CentreScaleTracker
import kotlinx.serialization.json.*
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.util.Random

/** Runs the tracker on a synthetic drive and writes tests/fixtures/tracker_parity.json for the Python port. */
class TrackerParityTest {
    private class Lamp(var cx:Float,var cy:Float,var size:Float,val vx:Float,val vy:Float,val growth:Float)

    @Test fun writeParityFixture() {
        val rng=Random(42)
        val tracker=CentreScaleTracker(RuntimeConfig())
        val lamps=mutableListOf<Lamp>()
        val frames=JsonArray((0 until 400).map { k ->
            if(rng.nextFloat()<0.08f) lamps+=Lamp(0.35f+0.3f*rng.nextFloat(),0.25f+0.1f*rng.nextFloat(),0.008f+0.01f*rng.nextFloat(),
                (rng.nextFloat()-0.5f)*0.12f,-0.03f*rng.nextFloat(),1.0f+0.15f*rng.nextFloat())
            val ms=k*200L+rng.nextInt(120)
            val boxes=mutableListOf<Box>()
            for(l in lamps) {
                l.cx+=l.vx*(1+l.cx-0.5f)*0.2f; l.cy+=l.vy*0.2f; l.size*=l.growth
                if(rng.nextFloat()<0.15f) continue                 // missed detection
                val j=0.004f*(rng.nextFloat()-0.5f)
                boxes+=Box(l.cx-l.size/2+j,l.cy-l.size/2,l.cx+l.size/2+j,l.cy+l.size/2)
            }
            lamps.removeAll { it.cx<0f || it.cx>1f || it.cy<0f || it.size>0.25f }
            if(rng.nextFloat()<0.1f) { val x=rng.nextFloat()*0.9f; val y=rng.nextFloat()*0.6f; boxes+=Box(x,y,x+0.01f,y+0.012f) }
            val detections=boxes.map { Detection(it,0.5f+0.5f*rng.nextFloat(),0,"lamp_head","yolo26_INT8") }
            val time=FrameTime(k.toLong(),ms,ms*1_000_000,ms*1_000_000,"sensor_realtime",0,1280,720)
            val result=tracker.update(detections,time)
            obj("t" to j(ms/1000.0),
                "boxes" to JsonArray(detections.map { d -> JsonArray(listOf(d.bbox.x1,d.bbox.y1,d.bbox.x2,d.bbox.y2).map { JsonPrimitive(it.toDouble()) }) }),
                "scores" to JsonArray(detections.map { JsonPrimitive(it.score.toDouble()) }),
                "matched_ids" to JsonArray(result.matched.map { JsonPrimitive(it.id) }))
        })
        val ids=frames.flatMap { it.jsonObject.getValue("matched_ids").jsonArray }.map { it.jsonPrimitive.long }.toSet()
        assertTrue(ids.size>20)
        val out=File("../../tests/fixtures")
        if(out.isDirectory) File(out,"tracker_parity.json").writeText(obj("source" to j("app/app/src/test/.../TrackerParityTest.kt"),"frames" to frames).toString())
    }
}
