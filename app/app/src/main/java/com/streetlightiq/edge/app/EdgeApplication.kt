package com.streetlightiq.edge.app

import android.app.Application
import android.content.Context
import androidx.room.Room
import com.streetlightiq.edge.camera.CameraController
import com.streetlightiq.edge.config.Preferences
import com.streetlightiq.edge.data.local.*
import com.streetlightiq.edge.data.remote.*
import com.streetlightiq.edge.location.*
import com.streetlightiq.edge.pipeline.EdgePipeline
import com.streetlightiq.edge.sync.*
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.HiltAndroidApp
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Inject
import javax.inject.Singleton

@HiltAndroidApp
class EdgeApplication:Application() { @Inject lateinit var syncEngine:SyncEngine }

@Module @InstallIn(SingletonComponent::class)
object EdgeModule {
    @Provides @Singleton fun database(@ApplicationContext context:Context)=Room.databaseBuilder(context,EdgeDatabase::class.java,"streetlightiq.db").build()
    @Provides @Singleton fun preferences(@ApplicationContext context:Context)=Preferences(context)
    @Provides @Singleton fun vault(@ApplicationContext context:Context)=TokenVault(context)
    @Provides @Singleton fun records(db:EdgeDatabase)=RecordRepository(db)
    @Provides @Singleton fun cloud(p:Preferences,v:TokenVault)=SupabaseCloud(p,v)
    @Provides @Singleton fun scheduler(@ApplicationContext context:Context)=SyncScheduler(context)
    @Provides @Singleton fun sync(records:RecordRepository,cloud:SupabaseCloud)=SyncEngine(records,cloud)
    @Provides @Singleton fun locations(@ApplicationContext context:Context):LocationProvider=AndroidLocationProvider(context)
    @Provides @Singleton fun camera(@ApplicationContext context:Context)=CameraController(context)
    @Provides @Singleton fun pipeline(@ApplicationContext context:Context,p:Preferences,r:RecordRepository,l:LocationProvider,v:TokenVault,s:SyncScheduler)=EdgePipeline(context,p,r,l,v,s)
}
