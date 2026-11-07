package com.streetlightiq.edge.inference
import com.streetlightiq.edge.config.ModelMetadata
import com.streetlightiq.edge.domain.*
import kotlin.math.*

object YoloDecoder {
    fun decode(m:ModelMetadata,n:Int,value:(Int,Int)->Float,fw:Int,fh:Int,scale:Float,px:Float,py:Float):List<Detection> {
        val items=ArrayList<Detection>()
        for(a in 0 until n) {
            val end=m.output_format=="end2end_xyxy"
            var score=if(end) value(4,a) else 0f
            var cls=if(end) value(5,a).takeIf { it.isFinite() }?.roundToInt() ?: -1 else -1
            if(!end) for(c in m.labels.indices) { val v=value(c+4,a); if(v>score) { score=v; cls=c } }
            if(!score.isFinite() || score<m.threshold || cls !in m.labels.indices) continue
            val fx=if(m.output_coordinate_units=="normalized") m.input_width.toFloat() else 1f
            val fy=if(m.output_coordinate_units=="normalized") m.input_height.toFloat() else 1f
            val x=value(0,a)*fx; val y=value(1,a)*fy; val w=value(2,a)*fx; val h=value(3,a)*fy
            if(!listOf(x,y,w,h).all { it.isFinite() }) continue
            val b=Box((((if(end) x else x-w/2)-px)/scale/fw).coerceIn(0f,1f),
                (((if(end) y else y-h/2)-py)/scale/fh).coerceIn(0f,1f),
                (((if(end) w else x+w/2)-px)/scale/fw).coerceIn(0f,1f),
                (((if(end) h else y+h/2)-py)/scale/fh).coerceIn(0f,1f))
            if(b.area>0) items+=Detection(b,score.coerceIn(0f,1f),cls,m.labels[cls],m.family+"_"+m.quantization)
        }
        return items.sortedByDescending { it.score }.take(if(m.output_format=="end2end_xyxy") 300 else 1000)
    }
}
