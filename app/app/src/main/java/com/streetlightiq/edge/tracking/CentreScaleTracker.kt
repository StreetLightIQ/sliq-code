package com.streetlightiq.edge.tracking

import com.streetlightiq.edge.domain.*
import com.streetlightiq.edge.config.RuntimeConfig
import java.util.UUID
import kotlin.math.*

interface Tracker {
    fun update(detections: List<Detection>, time: FrameTime): TrackingResult
    fun flush(): List<Track>
    fun reset()
}

/** Deterministic global association with centre/scale gates and smoothed velocity prediction. No learned ReID. */
class CentreScaleTracker(private val config: RuntimeConfig) : Tracker {
    private val tracks = linkedMapOf<Long,Track>()
    private var nextId = 1L
    private var previousNs = Long.MIN_VALUE
    override fun reset() { tracks.clear(); nextId=1; previousNs=Long.MIN_VALUE }
    override fun flush(): List<Track> = tracks.values.toList().also { tracks.clear() }
    override fun update(detections: List<Detection>, time: FrameTime): TrackingResult {
        require(time.monotonic_ns > previousNs) { "Frames must have increasing monotonic timestamps" }
        previousNs=time.monotonic_ns
        val completed=tracks.values.filter { time.monotonic_ns-it.lastTime.monotonic_ns > config.maxMissingMs*1_000_000 }
        completed.forEach { tracks.remove(it.id) }
        val existing=tracks.values.toList()
        val costs=existing.map { t ->
            DoubleArray(detections.size+existing.size) { col ->
                if(col >= detections.size) 1.0 else associationCost(t,detections[col],time)
            }
        }.toTypedArray()
        val assignment = if(existing.isEmpty()) IntArray(0) else Hungarian.assign(costs)
        val used=BooleanArray(detections.size)
        val matched=ArrayList<Track>()
        existing.forEachIndexed { row,t ->
            val col=assignment[row]
            if(col in detections.indices && costs[row][col]<1.0) {
                val d=detections[col]; used[col]=true
                val dt=max(1e-3f,(time.monotonic_ns-t.lastTime.monotonic_ns)/1e9f)
                t.vx=(0.6f*t.vx+0.4f*(d.bbox.cx-t.detection.bbox.cx)/dt).coerceIn(-1f,1f)
                t.vy=(0.6f*t.vy+0.4f*(d.bbox.cy-t.detection.bbox.cy)/dt).coerceIn(-1f,1f)
                t.detection=d; t.lastTime=time; t.matches++; t.misses=0
                t.scoreSum+=d.score; t.minScore=min(t.minScore,d.score); t.maxScore=max(t.maxScore,d.score)
                if(d.bbox.area>t.maxAreaBox.area) t.maxAreaBox=d.bbox
                matched+=t
            } else t.misses++
        }
        detections.forEachIndexed { i,d ->
            if(!used[i] && tracks.size<config.maxTracks) {
                val track=Track(nextId++,UUID.randomUUID().toString(),d,time,time)
                tracks[track.id]=track; matched+=track
            }
        }
        return TrackingResult(tracks.values.toList(),matched,completed)
    }
    private fun associationCost(t:Track,d:Detection,time:FrameTime):Double {
        if(d.class_id != t.detection.class_id || d.source != t.detection.source) return 1e6
        val old=t.detection.bbox; val b=d.bbox
        val dt=(time.monotonic_ns-t.lastTime.monotonic_ns)/1e9
        val distance=hypot(b.cx-old.cx-t.vx*dt,b.cy-old.cy-t.vy*dt)
        val gate=config.centreGate + min(0.08,dt*0.04)
        val scale=abs(ln(max(b.width,1e-5f)/max(old.width,1e-5f)))+abs(ln(max(b.height,1e-5f)/max(old.height,1e-5f)))
        if(distance>gate || scale>4) return 1e6
        return distance/gate*0.75 + scale*0.10
    }
}

/** Rectangular Hungarian minimization. Rows <= columns; dummy columns represent unmatched tracks. */
object Hungarian {
    fun assign(cost:Array<DoubleArray>):IntArray {
        if(cost.isEmpty()) return IntArray(0)
        val n=cost.size; val m=cost[0].size
        require(n<=m && cost.all { it.size==m && it.all(Double::isFinite) })
        val u=DoubleArray(n+1); val v=DoubleArray(m+1); val p=IntArray(m+1); val way=IntArray(m+1)
        for(i in 1..n) {
            p[0]=i; var j0=0
            val minv=DoubleArray(m+1) { Double.POSITIVE_INFINITY }; val used=BooleanArray(m+1)
            do {
                used[j0]=true; val i0=p[j0]; var delta=Double.POSITIVE_INFINITY; var j1=0
                for(j in 1..m) if(!used[j]) {
                    val cur=cost[i0-1][j-1]-u[i0]-v[j]
                    if(cur<minv[j]) { minv[j]=cur; way[j]=j0 }
                    if(minv[j]<delta) { delta=minv[j]; j1=j }
                }
                for(j in 0..m) { if(used[j]) { u[p[j]]+=delta; v[j]-=delta } else minv[j]-=delta }
                j0=j1
            } while(p[j0]!=0)
            do { val j1=way[j0]; p[j0]=p[j1]; j0=j1 } while(j0!=0)
        }
        return IntArray(n) { row -> (1..m).first { p[it]==row+1 }-1 }
    }
}
