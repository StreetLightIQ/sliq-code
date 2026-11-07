package com.streetlightiq.edge.config

import android.content.Context
import androidx.datastore.preferences.core.*
import androidx.datastore.preferences.preferencesDataStore
import com.streetlightiq.edge.domain.wireJson
import kotlinx.coroutines.flow.*
import kotlinx.serialization.Serializable

private val Context.edgePreferences by preferencesDataStore(name="edge_preferences")
@Serializable data class BackendConfig(val url:String="",val publicKey:String="")
@Serializable data class Settings(val backend:BackendConfig=BackendConfig(),val runtime:RuntimeConfig=RuntimeConfig())
class Preferences(private val context:Context) {
    private val key=stringPreferencesKey("settings")
    val settings:Flow<Settings> = context.edgePreferences.data.map { prefs ->
        prefs[key]?.let { runCatching { wireJson.decodeFromString<Settings>(it) }.getOrNull() } ?: Settings()
    }
    suspend fun get()=settings.first()
    suspend fun save(value:Settings) { context.edgePreferences.edit { it[key]=wireJson.encodeToString(Settings.serializer(),value) } }
}
