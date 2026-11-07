package com.streetlightiq.edge.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.activity.enableEdgeToEdge
import androidx.activity.SystemBarStyle
import androidx.core.content.ContextCompat
import com.streetlightiq.edge.ui.*
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class MainActivity:ComponentActivity() {
    private val model:EdgeViewModel by viewModels()
    override fun onCreate(savedInstanceState:Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge(SystemBarStyle.dark(android.graphics.Color.TRANSPARENT),SystemBarStyle.dark(android.graphics.Color.TRANSPARENT))
        setContent { EdgeApp(model) { active ->
            if(active) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            else window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        } }
    }
    override fun onStart() {
        super.onStart()
        if(ContextCompat.checkSelfPermission(this,Manifest.permission.ACCESS_FINE_LOCATION)==PackageManager.PERMISSION_GRANTED) model.beginLocation()
    }
    override fun onStop() { model.backgrounded(); super.onStop() }
}
