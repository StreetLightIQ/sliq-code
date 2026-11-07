package com.streetlightiq.edge

import com.streetlightiq.edge.config.ModelMetadata
import com.streetlightiq.edge.inference.YoloDecoder
import org.junit.Assert.*
import org.junit.Test

class YoloDecoderTest {
    private fun metadata(format:String="raw_xywh",units:String="pixels")=ModelMetadata(model_version="test",sha256="x",input_width=640,input_height=640,
        labels=listOf("streetlight","off"),output_format=format,output_coordinate_units=units)
    @Test fun rawBoxesUndoLetterboxToOriginalCoordinates() {
        // Original 640x480 frame is padded 80 pixels at top and bottom.
        val values=floatArrayOf(320f,320f,128f,96f,.8f,.1f)
        val found=YoloDecoder.decode(metadata(),1,{ c,_->values[c] },640,480,1f,0f,80f).single()
        assertEquals(.4f,found.bbox.x1,1e-6f); assertEquals(.4f,found.bbox.y1,1e-6f)
        assertEquals(.6f,found.bbox.x2,1e-6f); assertEquals(.6f,found.bbox.y2,1e-6f)
        assertEquals("streetlight",found.class_name)
    }
    @Test fun normalizedAndPixelBoxesHaveTheSameGeometry() {
        val pixels=floatArrayOf(320f,240f,64f,48f,.9f,.1f)
        val normalized=pixels.copyOf().apply { this[0]/=640; this[1]/=640; this[2]/=640; this[3]/=640 }
        val a=YoloDecoder.decode(metadata(),1,{c,_->pixels[c]},640,640,1f,0f,0f).single()
        val b=YoloDecoder.decode(metadata(units="normalized"),1,{c,_->normalized[c]},640,640,1f,0f,0f).single()
        assertEquals(a,b)
    }
    @Test fun endToEndUsesXyxyAndClassIdWithoutTreatingClassIdAsConfidence() {
        val rows=arrayOf(floatArrayOf(64f,128f,192f,256f,.9f,1f),floatArrayOf(0f,0f,2f,2f,.1f,0f))
        val d=YoloDecoder.decode(metadata("end2end_xyxy"),2,{c,a->rows[a][c]},640,640,1f,0f,0f).single()
        assertEquals("off",d.class_name); assertEquals(.1f,d.bbox.x1,1e-6f); assertEquals(.3f,d.bbox.x2,1e-6f)
    }
    @Test fun invalidScoresCoordinatesAndClassIdsNeverEscape() {
        val rows=arrayOf(floatArrayOf(0f,0f,64f,64f,Float.NaN,0f),floatArrayOf(Float.NaN,0f,64f,64f,.9f,0f),
            floatArrayOf(0f,0f,64f,64f,.9f,99f),floatArrayOf(50f,50f,20f,20f,.9f,0f))
        assertTrue(YoloDecoder.decode(metadata("end2end_xyxy"),rows.size,{c,a->rows[a][c]},640,640,1f,0f,0f).isEmpty())
    }
}
