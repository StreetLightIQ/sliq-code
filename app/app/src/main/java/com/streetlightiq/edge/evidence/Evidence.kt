package com.streetlightiq.edge.evidence

import android.graphics.Bitmap
import com.streetlightiq.edge.domain.*
import java.io.ByteArrayOutputStream
import kotlin.math.*

fun pixelRect(image:Bitmap,box:Box):IntArray {
    val x1=(box.x1*image.width).toInt().coerceIn(0,image.width-1)
    val y1=(box.y1*image.height).toInt().coerceIn(0,image.height-1)
    return intArrayOf(x1,y1,(box.x2*image.width).roundToInt().coerceIn(x1+1,image.width),(box.y2*image.height).roundToInt().coerceIn(y1+1,image.height))
}
data class SelectedEvidence(val jpeg:ByteArray?, val sceneJpeg:ByteArray?, val score:Double, val frame:FrameTime, val bbox:Box,
    val cropRect:List<Int>, val context:kotlinx.serialization.json.JsonObject,
    val location:LocationSample?, val gpsOffsetMs:Long?, val gpsQuality:String, val cropPng:ByteArray?=null)

/** One full-resolution frame of a track for the cloud stage. sizeLog = ln(sqrt(box area)), a distance proxy. */
data class PhysicsFrame(val jpeg:ByteArray, val frame:FrameTime, val bbox:Box, val score:Float, val sizeLog:Double,
    val context:kotlinx.serialization.json.JsonObject, val step:Int=0)

/** Up to [maxFrames] frames per track, one per box-size step of [stepFactor], so spread over distance.
 *  Over budget, the frame in the most crowded size gap is dropped (never the nearest or farthest). */
class PhysicsFramePicker(private val maxFrames:Int=5, stepFactor:Double=1.25) {
    private val step=ln(stepFactor)
    private val frames=mutableMapOf<Long,MutableList<PhysicsFrame>>()
    fun observe(id:Long,time:FrameTime,box:Box,score:Float,context:kotlinx.serialization.json.JsonObject,encode:()->ByteArray) {
        if(box.area<=0f) return
        val sizeLog=0.5*ln(box.area.toDouble())
        val bin=floor(sizeLog/step).toInt()
        val list=frames.getOrPut(id) { mutableListOf() }
        val same=list.indexOfFirst { it.step==bin }
        if(same>=0) {
            val old=list[same]
            val moreExtreme=list.size>1 && ((same==list.lastIndex && sizeLog>=old.sizeLog+step/2) || (same==0 && sizeLog<=old.sizeLog-step/2))
            if(!(score>old.score*1.10f || (moreExtreme && score>=old.score*0.90f))) return
            list[same]=PhysicsFrame(encode(),time,box,score,sizeLog,context,bin); return
        }
        list+=PhysicsFrame(encode(),time,box,score,sizeLog,context,bin); list.sortBy { it.sizeLog }
        if(list.size>maxFrames) {
            var drop=1; var gap=Double.MAX_VALUE
            for(i in 1 until list.size-1) { val g=list[i+1].sizeLog-list[i-1].sizeLog; if(g<gap) { gap=g; drop=i } }
            list.removeAt(drop)
        }
    }
    fun frames(id:Long):List<PhysicsFrame> = frames[id]?.toList() ?: emptyList()
    fun discard(id:Long) { frames.remove(id) }
}
fun fullFrameJpeg(image:Bitmap,quality:Int=90):ByteArray=
    ByteArrayOutputStream().use { out -> image.compress(Bitmap.CompressFormat.JPEG,quality,out); out.toByteArray() }

interface EvidenceSelector {
    fun observe(track:Track,image:Bitmap,time:FrameTime,context:kotlinx.serialization.json.JsonObject,location:LocationSample?,offset:Long?,gpsQuality:String)
    fun best(id:Long):SelectedEvidence?
    fun discard(id:Long)
}
class BestCropSelector(private val saveImages:Boolean,private val saveContextImages:Boolean=true):EvidenceSelector {
    private val selected=mutableMapOf<Long,SelectedEvidence>()
    override fun observe(track:Track,image:Bitmap,time:FrameTime,context:kotlinx.serialization.json.JsonObject,location:LocationSample?,offset:Long?,gpsQuality:String) {
        val box=track.detection.bbox
        val score=track.detection.score*sqrt(box.area.toDouble())
        if(selected[track.id]?.let { score <= it.score*1.10 } == true) return
        val rect=pixelRect(image,box.expanded(max(box.width,box.height)*0.4f))
        val jpeg=if(saveImages) {
            val crop=Bitmap.createBitmap(image,rect[0],rect[1],rect[2]-rect[0],rect[3]-rect[1])
            val scale=min(1.0,320.0/max(crop.width,crop.height))
            val small=Bitmap.createScaledBitmap(crop,max(1,(crop.width*scale).toInt()),max(1,(crop.height*scale).toInt()),true)
            val bytes=ByteArrayOutputStream().use { out -> small.compress(Bitmap.CompressFormat.JPEG,82,out); out.toByteArray() }
            if(small !== crop) small.recycle(); if(crop !== image) crop.recycle(); bytes
        } else null
        val sceneJpeg=if(saveContextImages) ByteArrayOutputStream().use { out ->
            val factor=min(1.0,960.0/max(image.width,image.height))
            val small=Bitmap.createScaledBitmap(image,max(1,(image.width*factor).toInt()),max(1,(image.height*factor).toInt()),true)
            small.compress(Bitmap.CompressFormat.JPEG,80,out)
            if(small !== image) small.recycle()
            out.toByteArray()
        } else null
        val cropPng=if(saveImages) {
            val crop=Bitmap.createBitmap(image,rect[0],rect[1],rect[2]-rect[0],rect[3]-rect[1])
            try { ByteArrayOutputStream().use { out -> crop.compress(Bitmap.CompressFormat.PNG,100,out); out.toByteArray() } }
            finally { if(crop!==image) crop.recycle() }
        } else null
        selected[track.id]=SelectedEvidence(jpeg,sceneJpeg,score,time,box,rect.toList(),context,location,offset,gpsQuality,cropPng)
    }
    override fun best(id:Long)=selected[id]
    override fun discard(id:Long) { selected.remove(id) }
}
