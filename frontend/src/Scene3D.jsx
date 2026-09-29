import { useEffect, useRef } from 'react'
import * as THREE from 'three'

/**
 * A lightweight Three.js "digital twin" of the microgrid site.
 * Purely visual — it does not compute physics itself. The parent component
 * (App.jsx) derives the actual sensor numbers (irradiance, wind speed) from
 * the same sliders and sends those to the backend; this component just
 * renders what those conditions would look like on site.
 *
 * Props:
 *   timeOfDay   - 0..24 (hours), drives sun position + sky color
 *   cloudCover  - 0..100 (%), drives cloud density/opacity + light dimming
 *   windSpeed   - 0..100 (km/h), drives turbine blade rotation speed
 */
export default function Scene3D({ timeOfDay, cloudCover, windSpeed }) {
  const mountRef = useRef(null)
  const stateRef = useRef({}) // holds three.js objects across renders without re-triggering React

  // ---- one-time scene setup ----
  useEffect(() => {
    const mount = mountRef.current
    const width = mount.clientWidth
    const height = mount.clientHeight

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000)
    camera.position.set(14, 9, 16)
    camera.lookAt(0, 2, 0)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(width, height)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    mount.appendChild(renderer.domElement)

    // Ground
    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(60, 60),
      new THREE.MeshStandardMaterial({ color: 0x3c5a3c, roughness: 1 })
    )
    ground.rotation.x = -Math.PI / 2
    scene.add(ground)

    // --- Wind turbine ---
    const turbine = new THREE.Group()
    const tower = new THREE.Mesh(
      new THREE.CylinderGeometry(0.15, 0.3, 6, 12),
      new THREE.MeshStandardMaterial({ color: 0xe6e9ec })
    )
    tower.position.y = 3
    turbine.add(tower)

    const nacelle = new THREE.Mesh(
      new THREE.BoxGeometry(0.5, 0.5, 1.2),
      new THREE.MeshStandardMaterial({ color: 0xcfd4d9 })
    )
    nacelle.position.set(0, 6, 0.3)
    turbine.add(nacelle)

    const rotor = new THREE.Group()
    rotor.position.set(0, 6, 0.9)
    for (let i = 0; i < 3; i++) {
      const blade = new THREE.Mesh(
        new THREE.BoxGeometry(0.12, 2.6, 0.05),
        new THREE.MeshStandardMaterial({ color: 0xf4f6f8 })
      )
      blade.position.y = 1.3
      const holder = new THREE.Group()
      holder.rotation.z = (i * Math.PI * 2) / 3
      holder.add(blade)
      rotor.add(holder)
    }
    turbine.add(rotor)
    turbine.position.set(-5, 0, -2)
    scene.add(turbine)

    // --- Solar array ---
    const solarGroup = new THREE.Group()
    const panelMat = new THREE.MeshStandardMaterial({
      color: 0x1a2e4a, emissive: 0x223355, emissiveIntensity: 0.3, roughness: 0.4,
    })
    for (let row = 0; row < 2; row++) {
      for (let col = 0; col < 4; col++) {
        const panel = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.08, 1.0), panelMat.clone())
        panel.position.set(2 + col * 1.9, 0.6, -1 + row * 1.3)
        panel.rotation.x = -0.35
        solarGroup.add(panel)
      }
    }
    scene.add(solarGroup)

    // --- Sun + lighting ---
    const sunMesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.6, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0xffe28a })
    )
    scene.add(sunMesh)

    const sunLight = new THREE.DirectionalLight(0xffffff, 1)
    scene.add(sunLight)
    const ambient = new THREE.AmbientLight(0x404060, 0.6)
    scene.add(ambient)

    // --- Clouds (pool, visibility/opacity driven by cloudCover prop) ---
    const cloudPool = []
    const cloudMat = new THREE.MeshStandardMaterial({ color: 0xffffff, transparent: true, opacity: 0.85 })
    for (let i = 0; i < 10; i++) {
      const group = new THREE.Group()
      const puffCount = 3 + (i % 3)
      for (let j = 0; j < puffCount; j++) {
        const puff = new THREE.Mesh(new THREE.SphereGeometry(0.9 + Math.random() * 0.5, 8, 8), cloudMat)
        puff.position.set(j * 0.9 - puffCount * 0.4, Math.random() * 0.3, Math.random() * 0.5)
        group.add(puff)
      }
      const angle = (i / 10) * Math.PI * 2
      group.position.set(Math.cos(angle) * 10, 7 + Math.random() * 1.5, Math.sin(angle) * 10 - 3)
      group.visible = false
      scene.add(group)
      cloudPool.push(group)
    }

    // Resize handling
    const resize = () => {
      const w = mount.clientWidth, h = mount.clientHeight
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    }
    const ro = new ResizeObserver(resize)
    ro.observe(mount)

    let animId
    let lastT = performance.now()
    const animate = (t) => {
      const dt = (t - lastT) / 1000
      lastT = t
      const s = stateRef.current
      const angularSpeed = ((s.windSpeed ?? 0) / 100) * 6 // rad/sec at max wind
      rotor.rotation.z += angularSpeed * dt
      renderer.render(scene, camera)
      animId = requestAnimationFrame(animate)
    }
    animId = requestAnimationFrame(animate)

    stateRef.current = {
      scene, camera, renderer, sunMesh, sunLight, ambient, cloudPool, solarGroup,
      windSpeed, timeOfDay, cloudCover,
    }

    return () => {
      cancelAnimationFrame(animId)
      ro.disconnect()
      mount.removeChild(renderer.domElement)
      renderer.dispose()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ---- reactive updates when sliders change ----
  useEffect(() => {
    const s = stateRef.current
    if (!s.scene) return
    s.windSpeed = windSpeed // read by animate() loop via closure over stateRef

    // Sun position from time-of-day (0-24h)
    const elevation = Math.max(-0.15, Math.sin(((timeOfDay - 6) / 12) * Math.PI))
    const azimuth = (timeOfDay / 24) * Math.PI * 2
    const radius = 20
    const sunX = Math.cos(azimuth) * radius
    const sunY = Math.max(1, elevation * radius)
    const sunZ = Math.sin(azimuth) * radius * 0.4
    s.sunMesh.position.set(sunX, sunY, sunZ)
    s.sunMesh.visible = elevation > -0.05

    const cloudAtten = 1 - (cloudCover / 100) * 0.8
    const lightStrength = Math.max(0.05, elevation) * cloudAtten
    s.sunLight.position.set(sunX, sunY, sunZ)
    s.sunLight.intensity = lightStrength * 1.4
    s.ambient.intensity = 0.25 + (1 - Math.max(0, elevation)) * 0.25

    // Sky color: night blue -> day blue, dimmed by cloud cover
    const dayColor = new THREE.Color(0x8fc2e8)
    const nightColor = new THREE.Color(0x0a1020)
    const dayFactor = Math.max(0, Math.min(1, (elevation + 0.15) / 1.0))
    const sky = nightColor.clone().lerp(dayColor, dayFactor)
    sky.multiplyScalar(1 - (cloudCover / 100) * 0.35)
    s.scene.background = sky

    // Solar panel glow responds to irradiance-ish (elevation * cloud attenuation)
    const glow = Math.max(0.1, elevation) * cloudAtten
    s.solarGroup.children.forEach((panel) => {
      panel.material.emissiveIntensity = 0.15 + glow * 1.2
    })

    // Cloud pool visibility/opacity scale with cloudCover
    const visibleCount = Math.round((cloudCover / 100) * s.cloudPool.length)
    s.cloudPool.forEach((group, i) => {
      group.visible = i < visibleCount
      group.children.forEach((puff) => {
        puff.material.opacity = 0.5 + (cloudCover / 100) * 0.45
      })
    })
  }, [timeOfDay, cloudCover, windSpeed])

  return <div ref={mountRef} style={{ width: '100%', height: '100%' }} />
}
