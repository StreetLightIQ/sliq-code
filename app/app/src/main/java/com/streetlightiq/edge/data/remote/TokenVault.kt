package com.streetlightiq.edge.data.remote

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import com.streetlightiq.edge.domain.wireJson
import kotlinx.serialization.Serializable
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

@Serializable data class CloudSession(val userId:String,val email:String,val projectUrl:String,
    val accessToken:String,val refreshToken:String,val expiresAtMs:Long)
class TokenVault(context:Context) {
    private val preferences=context.getSharedPreferences("cloud_tokens",Context.MODE_PRIVATE)
    private val alias="streetlightiq-auth-v1"
    private fun key():SecretKey {
        val store=KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        (store.getKey(alias,null) as? SecretKey)?.let { return it }
        return KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES,"AndroidKeyStore").apply {
            init(KeyGenParameterSpec.Builder(alias,KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM).setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE).build())
        }.generateKey()
    }
    @Synchronized fun read():CloudSession? {
        val text=preferences.getString("session",null) ?: return null
        return runCatching {
            val packed=Base64.decode(text,Base64.NO_WRAP)
            val iv=packed.copyOfRange(0,12); val encrypted=packed.copyOfRange(12,packed.size)
            val cipher=Cipher.getInstance("AES/GCM/NoPadding").apply { init(Cipher.DECRYPT_MODE,key(),GCMParameterSpec(128,iv)) }
            wireJson.decodeFromString<CloudSession>(cipher.doFinal(encrypted).toString(Charsets.UTF_8))
        }.getOrNull()
    }
    @Synchronized fun save(value:CloudSession) {
        val cipher=Cipher.getInstance("AES/GCM/NoPadding").apply { init(Cipher.ENCRYPT_MODE,key()) }
        val bytes=wireJson.encodeToString(CloudSession.serializer(),value).toByteArray()
        preferences.edit().putString("session",Base64.encodeToString(cipher.iv+cipher.doFinal(bytes),Base64.NO_WRAP)).commit()
    }
    @Synchronized fun clear() { preferences.edit().remove("session").commit() }
}
