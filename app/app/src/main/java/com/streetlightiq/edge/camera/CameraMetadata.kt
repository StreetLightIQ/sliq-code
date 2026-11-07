package com.streetlightiq.edge.camera

import android.hardware.camera2.*
import android.hardware.camera2.params.*
import android.graphics.*
import android.util.*
import com.streetlightiq.edge.domain.*
import kotlinx.serialization.json.*

/** Preserve every accessible public/vendor key, including missing-value and encoding information. */
object CameraMetadata {
    fun value(v:Any?):JsonElement=when(v) {
        null -> JsonNull
        is Float -> if(v.isFinite()) j(v) else JsonNull
        is Double -> if(v.isFinite()) j(v) else JsonNull
        is Rational -> obj("numerator" to j(v.numerator),"denominator" to j(v.denominator))  // must precede Number: Rational extends Number
        is Number -> j(v)
        is Boolean -> j(v)
        is String -> j(v)
        is IntArray -> JsonArray(v.map { j(it) })
        is LongArray -> JsonArray(v.map { j(it) })
        is FloatArray -> JsonArray(v.map { value(it) })
        is DoubleArray -> JsonArray(v.map { value(it) })
        is ByteArray -> JsonArray(v.map { j(it.toInt() and 255) })
        is BooleanArray -> JsonArray(v.map { j(it) })
        is Array<*> -> JsonArray(v.map { value(it) })
        is Range<*> -> obj("lower" to value(v.lower),"upper" to value(v.upper))
        is android.util.Pair<*,*> -> JsonArray(listOf(value(v.first),value(v.second)))
        is Rect -> JsonArray(listOf(v.left,v.top,v.right,v.bottom).map { j(it) })
        is PointF -> JsonArray(listOf(value(v.x),value(v.y)))
        is Size -> JsonArray(listOf(j(v.width),j(v.height)))
        is SizeF -> JsonArray(listOf(j(v.width),j(v.height)))
        is RggbChannelVector -> JsonArray(listOf(v.red,v.greenEven,v.greenOdd,v.blue).map { value(it) })
        is ColorSpaceTransform -> JsonArray((0..2).flatMap { row -> (0..2).map { col -> value(v.getElement(col,row)) } })
        is BlackLevelPattern -> JsonArray((0..1).flatMap { row -> (0..1).map { col -> j(v.getOffsetForIndex(col,row)) } })
        is TonemapCurve -> obj("red" to curve(v,0),"green" to curve(v,1),"blue" to curve(v,2))
        is LensShadingMap -> obj("rows" to j(v.rowCount),"columns" to j(v.columnCount),
            "gains_rg_gb_b" to JsonArray((0 until v.rowCount).flatMap { row -> (0 until v.columnCount).flatMap { col -> (0..3).map { c -> value(v.getGainFactor(c,col,row)) } } }))
        else -> obj("encoding" to j("android_toString"),"type" to j(v.javaClass.name),"value" to j(v.toString()))
    }
    private fun curve(v:TonemapCurve,c:Int):JsonArray=JsonArray((0 until v.getPointCount(c)).map { i -> value(v.getPoint(c,i)) })
    fun capture(r:TotalCaptureResult):JsonObject=JsonObject(r.keys.associate { key -> key.name to runCatching { value(r.get(key)) }.getOrElse { JsonNull } })
    fun characteristics(c:CameraCharacteristics):JsonObject=JsonObject(c.keys.associate { key -> key.name to runCatching { value(c.get(key)) }.getOrElse { JsonNull } })
}
