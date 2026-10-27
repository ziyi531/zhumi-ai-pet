// 团子表情引擎：表情参数、插值动画和五种画风（发光大眼、软萌团子、猫猫脸、呆猫八条风、像素复古）。
// 表情预览页 faces.html 和网页宠物 pet.html 共用。固件 ai_pet_fw/main/face 里的 C++ 画法照着这里的 bajo 移植，改表情时两边一起改。
// ================= 表情参数：每个表情只是一组数字，切换时逐帧插值，所以会“变形”过去 =================
const FACE_LIST = [
  ['idle', '待机'], ['listen', '倾听'], ['think', '思考'], ['happy', '开心'],
  ['shy', '害羞'], ['pout', '傲娇哼'], ['sad', '难过'], ['lonely', '孤单'],
  ['curious', '好奇'], ['sleepy', '犯困'], ['sleep', '睡着'], ['peek', '探头'],
];
const BASE = { eyeW: 1, eyeH: 1, lidT: 0, lidB: 0, brow: 0, lookX: 0, lookY: 0, eyeY: 0, scaleL: 1, scaleR: 1,
               blush: 0.15, ear: 0, pupil: 0.35, mode: 'open', mouth: 'w', extra: null, peek: false, lookAnim: false };
const FACE_DEFS = {
  idle: {},
  listen: { eyeW: 1.06, eyeH: 1.12, lookY: -4, mouth: 'o', extra: 'waves', ear: 1, pupil: 0.8 },
  think: { lookY: -10, lidB: 0.15, mouth: 'flat', extra: 'dots', lookAnim: true, pupil: 0.3 },
  happy: { mode: 'happy', mouth: 'smile', blush: 0.9, extra: 'sparkle', ear: 0.4 },
  shy: { lookX: -12, lookY: 10, lidT: 0.18, eyeH: 0.92, mouth: 'flat', blush: 1, ear: -0.3, pupil: 0.6 },
  pout: { mode: 'pout', mouth: 'frown', blush: 0.4, extra: 'anger', ear: -0.7 },
  sad: { brow: 0.4, lookY: 8, lidT: 0.08, mouth: 'frown', extra: 'tear', blush: 0, ear: -1, pupil: 0.8 },
  lonely: { brow: 0.3, lookX: -10, lookY: 12, lidT: 0.28, mouth: 'flat', blush: 0, ear: -0.8, pupil: 0.7 },
  curious: { scaleL: 1.15, scaleR: 0.82, lookX: 6, mouth: 'o', extra: 'q', ear: 0.6, pupil: 0.9 },
  sleepy: { lidT: 0.55, eyeY: 6, mouth: 'flat', extra: 'z', blush: 0.1, ear: -0.5, pupil: 0.2 },
  sleep: { mode: 'closed', eyeY: 8, mouth: 'flat', extra: 'zzz', blush: 0.1, ear: -0.6 },
  peek: { lookY: -10, extra: '!', peek: true, mouth: 'none', ear: 0.8, pupil: 0.9 },
};
const NUM = ['eyeW', 'eyeH', 'lidT', 'lidB', 'brow', 'lookX', 'lookY', 'eyeY', 'scaleL', 'scaleR', 'blush', 'ear', 'pupil'];
const BLINKABLE = new Set(['idle', 'listen', 'shy', 'lonely', 'curious', 'peek', 'sad', 'think']);

function goalOf(face) { return Object.assign({}, BASE, FACE_DEFS[face]); }
function newState(face) {
  const tg = goalOf(face), now = performance.now();
  return { face, tg, p: { ...tg }, start: now, blinkAt: now + 1200 + Math.random() * 3000, blinkStart: -1e9,
           glance: 0, glanceUntil: 0, override: null, overrideUntil: 0 };
}
function setFace(st, face) {
  st.face = face; st.tg = goalOf(face); st.start = performance.now();
  Object.assign(st.p, { mode: st.tg.mode, mouth: st.tg.mouth, extra: st.tg.extra, peek: st.tg.peek });
}
const easeOut = x => 1 - Math.pow(1 - x, 3);

function step(st, t) {
  const tg = st.tg, p = st.p, age = t - st.start;
  let lx = tg.lookX, ey = tg.eyeY;
  if (tg.lookAnim) lx = Math.sin(t / 380) * 14;
  if (st.face === 'idle') {   // 待机时偶尔东张西望
    if (t > st.glanceUntil) { st.glance = Math.random() < 0.5 ? 0 : (Math.random() < 0.5 ? -12 : 12); st.glanceUntil = t + 1500 + Math.random() * 2500; }
    lx += st.glance;
  }
  if (tg.peek) ey = 140 * (1 - easeOut(Math.min(1, age / 1100))) + 50;
  const goal = { ...tg, lookX: lx, eyeY: ey };
  for (const k of NUM) p[k] += (goal[k] - p[k]) * (k === 'eyeY' && tg.peek ? 1 : 0.18);
  let blink = 0;
  if (p.mode === 'open' && BLINKABLE.has(st.face)) {
    if (t > st.blinkAt) { st.blinkStart = t; st.blinkAt = t + 2200 + Math.random() * 4000; }
    const d = t - st.blinkStart;
    if (d < 170) blink = 1 - Math.abs(d / 85 - 1);
  }
  return { t, blink, age, face: st.face };
}

// ================= 通用画法 =================
function eyeBox(G, p, side) {
  const s = side < 0 ? p.scaleL : p.scaleR;
  return { cx: 160 + side * G.gap + p.lookX, cy: G.cy + p.eyeY + p.lookY, w: G.w * p.eyeW * s, h: G.h * p.eyeH * s };
}
// 眼皮：用裁剪区域把眼睛上下盖住，brow>0 外眼角下垂（难过），brow<0 内眼角下压（生气）
function lidY(e, p, side, lt, x) {
  const u = Math.min(1, Math.max(0, (x - (e.cx - e.w / 2)) / e.w));
  const outer = side < 0 ? 1 - u : u;
  const slope = p.brow > 0 ? p.brow * outer : -p.brow * (1 - outer);
  return e.cy - e.h / 2 + lt * e.h + slope * e.h * 0.7;
}
function withLids(ctx, e, p, side, blink, fn) {
  const top = e.cy - e.h / 2, bot = e.cy + e.h / 2, lt = Math.max(p.lidT, blink), lb = p.lidB;
  const yTop = x => (lt < 0.01 && Math.abs(p.brow) < 0.01) ? top - 40 : lidY(e, p, side, lt, x);
  const yBot = lb > 0.01 ? bot - lb * e.h : bot + 40;
  const x0 = e.cx - e.w / 2 - 40, x1 = e.cx + e.w / 2 + 40, xm0 = e.cx - e.w / 2, xm1 = e.cx + e.w / 2;
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(x0, yTop(x0)); ctx.lineTo(xm0, yTop(xm0)); ctx.lineTo(xm1, yTop(xm1)); ctx.lineTo(x1, yTop(x1));
  ctx.lineTo(x1, yBot); ctx.lineTo(x0, yBot); ctx.closePath();
  ctx.clip(); fn(); ctx.restore();
}
function strokeEye(ctx, e, mode, side) {
  ctx.beginPath();
  if (mode === 'happy') ctx.arc(e.cx, e.cy + e.h * 0.2, e.w * 0.5, Math.PI * 1.12, Math.PI * 1.88);
  else if (mode === 'closed') ctx.arc(e.cx, e.cy - e.w * 0.12, e.w * 0.5, Math.PI * 0.15, Math.PI * 0.85);
  else if (mode === 'pout') {
    const d = -side;  // 左眼“>”、右眼“<”，尖角朝内
    ctx.moveTo(e.cx - d * e.w * 0.3, e.cy - e.h * 0.26); ctx.lineTo(e.cx + d * e.w * 0.3, e.cy); ctx.lineTo(e.cx - d * e.w * 0.3, e.cy + e.h * 0.26);
  }
  ctx.stroke();
}
function drawEye(ctx, S, G, p, o, side) {
  const e = eyeBox(G, p, side);
  if (p.mode === 'open' && o.blink < 0.8) withLids(ctx, e, p, side, o.blink, () => S.openEye(ctx, e, p, side, o));
  else strokeEye(ctx, e, p.mode === 'open' ? 'closed' : p.mode, side);
}
function drawMouth(ctx, type, x, y, s, fill) {
  ctx.beginPath();
  if (type === 'w') { ctx.arc(x - s * 0.5, y, s * 0.5, 0, Math.PI); ctx.moveTo(x + s, y); ctx.arc(x + s * 0.5, y, s * 0.5, 0, Math.PI); ctx.stroke(); }
  else if (type === 'smile') {
    ctx.arc(x, y - s * 0.3, s * 1.0, Math.PI * 0.12, Math.PI * 0.88);
    if (fill) { ctx.closePath(); ctx.save(); ctx.fillStyle = fill; ctx.fill(); ctx.restore(); }
    ctx.stroke();
  }
  else if (type === 'o') { ctx.ellipse(x, y + 2, s * 0.42, s * 0.52, 0, 0, Math.PI * 2); if (fill) { ctx.save(); ctx.fillStyle = fill; ctx.fill(); ctx.restore(); } ctx.stroke(); }
  else if (type === 'frown') { ctx.arc(x, y + s * 1.1, s * 0.95, Math.PI * 1.25, Math.PI * 1.75); ctx.stroke(); }
  else if (type === 'flat') { ctx.moveTo(x - s * 0.55, y); ctx.lineTo(x + s * 0.55, y); ctx.stroke(); }
}
function drawBlush(ctx, G, p, color, lines) {
  if (p.blush < 0.05) return;
  for (const side of [-1, 1]) {
    const x = 160 + side * (G.gap + 14), y = G.cy + p.eyeY + G.h * 0.62;
    ctx.save(); ctx.globalAlpha = Math.min(1, p.blush); ctx.fillStyle = color;
    ctx.beginPath(); ctx.ellipse(x, y, 20, 10, 0, 0, Math.PI * 2); ctx.fill();
    if (lines && p.blush > 0.7) {
      ctx.strokeStyle = '#ff6f8f'; ctx.lineWidth = 2.5; ctx.lineCap = 'round';
      for (let i = -1; i <= 1; i++) { ctx.beginPath(); ctx.moveTo(x + i * 9 - 3, y + 4); ctx.lineTo(x + i * 9 + 3, y - 4); ctx.stroke(); }
    }
    ctx.restore();
  }
}
function star(ctx, x, y, r) {
  ctx.beginPath();
  for (let i = 0; i < 8; i++) { const a = i * Math.PI / 4, rr = i % 2 ? r * 0.35 : r; ctx.lineTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr); }
  ctx.closePath(); ctx.fill();
}
function drawExtras(ctx, G, p, o, c) {
  const t = o.t;
  // 角标位置可以按风格覆盖（猫猫脸的右上角被耳朵占了）
  const P = Object.assign({ z: [232, 82], q: [256, 64], dots: [244, 56], anger: [262, 52] }, c.pos);
  ctx.save(); ctx.fillStyle = c.extra; ctx.strokeStyle = c.extra; ctx.lineCap = 'round';
  switch (p.extra) {
    case 'zzz':
      for (let i = 0; i < 3; i++) {
        const ph = ((t / 1600) + i / 3) % 1;
        ctx.globalAlpha = Math.min(1, (1 - ph) * 1.4); ctx.font = `bold ${16 + i * 7}px sans-serif`;
        ctx.fillText('Z', P.z[0] + i * 18 + ph * 8, P.z[1] - i * 20 - ph * 22);
      }
      break;
    case 'z': ctx.globalAlpha = 0.55 + 0.45 * Math.sin(t / 500); ctx.font = 'bold 22px sans-serif'; ctx.fillText('z', P.z[0] + 16, P.z[1] - 20 - (t / 120) % 8); break;
    case 'q': ctx.font = 'bold 46px sans-serif'; ctx.fillStyle = c.q || c.extra; ctx.fillText('?', P.q[0], P.q[1] + Math.sin(t / 300) * 4); break;
    case '!': if (o.age > 1000) { ctx.font = 'bold 46px sans-serif'; ctx.fillStyle = c.q || c.extra; ctx.fillText('!', P.q[0] + 10, P.q[1]); } break;
    case 'tear': {
      const ph = (t / 1500) % 1, x = 160 - G.gap - G.w * 0.32 + p.lookX, y = G.cy + p.eyeY + G.h * 0.35 + ph * 46;
      ctx.globalAlpha = 1 - ph; ctx.fillStyle = c.tear;
      ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI); ctx.lineTo(x, y - 13); ctx.closePath(); ctx.fill(); break;
    }
    case 'dots': { const n = Math.floor(t / 350) % 4; for (let i = 0; i < n; i++) { ctx.beginPath(); ctx.arc(P.dots[0] + i * 15, P.dots[1], 5, 0, Math.PI * 2); ctx.fill(); } break; }
    case 'sparkle':
      ctx.fillStyle = c.sparkle;
      [[62, 54], [258, 44], [272, 150], [48, 150]].forEach(([x, y], i) => star(ctx, x, y, 6 + 5 * Math.max(0, Math.sin(t / 260 + i * 1.7))));
      break;
    case 'anger': {
      ctx.strokeStyle = c.anger; ctx.lineWidth = 5; const [x, y] = P.anger, k = 1 + 0.12 * Math.sin(t / 150);
      for (let i = 0; i < 4; i++) { const a = i * Math.PI / 2 + Math.PI / 4; ctx.beginPath(); ctx.arc(x + Math.cos(a) * 13 * k, y + Math.sin(a) * 13 * k, 8 * k, a + Math.PI * 0.75, a + Math.PI * 1.25); ctx.stroke(); }
      break;
    }
    case 'waves':
      ctx.lineWidth = 4;
      for (const side of [-1, 1]) for (let i = 0; i < 2; i++) {
        const ph = ((t / 900) + i / 2) % 1; ctx.globalAlpha = 1 - ph;
        const r = 10 + ph * 22, cx = 160 + side * 122, cy = G.cy + p.eyeY;
        ctx.beginPath(); side > 0 ? ctx.arc(cx, cy, r, -Math.PI * 0.3, Math.PI * 0.3) : ctx.arc(cx, cy, r, Math.PI * 0.7, Math.PI * 1.3); ctx.stroke();
      }
      break;
  }
  ctx.restore();
}
// 探头：眼睛从“桌沿”后面冒出来，两只小爪子扒着边
function drawLedge(ctx, p, o, c) {
  if (!p.peek) return;
  ctx.save(); ctx.shadowBlur = 0;
  ctx.fillStyle = c.ledge; ctx.fillRect(0, 182, 320, 58);
  ctx.fillStyle = c.ledgeTop; ctx.fillRect(0, 182, 320, 5);
  if (o.age > 500) {
    ctx.fillStyle = c.paw; ctx.strokeStyle = c.pawLine; ctx.lineWidth = 2.5;
    for (const x of [62, 258]) {
      ctx.beginPath(); ctx.ellipse(x, 186, 18, 12, 0, Math.PI, 0); ctx.closePath(); ctx.fill(); ctx.stroke();
      for (const dx of [-6, 0, 6]) { ctx.beginPath(); ctx.moveTo(x + dx, 186); ctx.lineTo(x + dx, 180); ctx.stroke(); }  // 爪缝
    }
  }
  ctx.restore();
}
function drawHearts(ctx, hearts, t, color) {
  ctx.save(); ctx.fillStyle = color;
  for (const h of hearts) {
    const ph = (t - h.t) / 1200; if (ph > 1) continue;
    ctx.globalAlpha = 1 - ph; const x = h.x, y = h.y - ph * 40, s = 9;
    ctx.beginPath(); ctx.moveTo(x, y + s);
    ctx.bezierCurveTo(x - s * 1.6, y - s * 0.2, x - s * 0.7, y - s * 1.4, x, y - s * 0.4);
    ctx.bezierCurveTo(x + s * 0.7, y - s * 1.4, x + s * 1.6, y - s * 0.2, x, y + s);
    ctx.fill();
  }
  ctx.restore();
}

// ================= 四种风格 =================
const STYLES = {
  glow: {
    name: '发光大眼', desc: '黑底、两只会发光的大眼睛，没有多余五官',
    note: '<b>优点</b>：最有“电子生命”的感觉，夜里很好看；纯黑背景在这块屏上也最省电（背光可以调暗）。<br><b>注意</b>：发光效果在 ESP32 上要用预渲染的光晕图层来做，画面会比这里略朴素一点。',
    G: { gap: 56, cy: 104, w: 62, h: 80 },
    c: { extra: '#62f0ff', q: '#ffd166', tear: '#7fb8ff', sparkle: '#ffe680', anger: '#ff6b6b', ledge: '#0c1118', ledgeTop: '#1d2a3a', paw: '#0c1118', pawLine: '#62f0ff', heart: '#ff7eb6' },
    openEye(ctx, e) { ctx.beginPath(); ctx.roundRect(e.cx - e.w / 2, e.cy - e.h / 2, e.w, e.h, Math.min(e.w, e.h) * 0.32); ctx.fill(); },
    draw(ctx, p, o) {
      const G = this.G, C = '#62f0ff';
      ctx.fillStyle = '#04070b'; ctx.fillRect(0, 0, 320, 240);
      ctx.translate(0, Math.sin(o.t / 1300) * 2);
      ctx.save(); ctx.shadowColor = C; ctx.shadowBlur = 22; ctx.fillStyle = C; ctx.strokeStyle = C; ctx.lineWidth = 12; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      drawEye(ctx, this, G, p, o, -1); drawEye(ctx, this, G, p, o, 1);
      if (p.mouth !== 'w' && p.mouth !== 'none') { ctx.lineWidth = 5; ctx.globalAlpha = 0.85; drawMouth(ctx, p.mouth, 160, G.cy + p.eyeY + 66, 12); }
      ctx.restore();
      // 黑底上淡淡的腮红会变成暗红色斑，所以只在害羞、开心这类表情才亮
      ctx.save(); ctx.shadowColor = '#ff6fae'; ctx.shadowBlur = 18; drawBlush(ctx, G, { ...p, blush: Math.max(0, p.blush - 0.35) }, '#ff6fae'); ctx.restore();
      ctx.save(); ctx.shadowColor = C; ctx.shadowBlur = 12; drawExtras(ctx, G, p, o, this.c); ctx.restore();
      drawLedge(ctx, p, o, this.c);
    },
  },
  dango: {
    name: '软萌团子', desc: '奶油底色、黑豆眼加高光、腮红和 ω 小嘴',
    note: '<b>优点</b>：最“可爱系”，和粉白外壳、毛绒外套都很搭；表情细节最丰富。<br><b>注意</b>：浅色背景整屏都要亮，比黑底费电一些；夜间模式可以换成深色底。',
    G: { gap: 58, cy: 106, w: 50, h: 60 },
    c: { extra: '#8a9cff', q: '#ff9f43', tear: '#7fb8ff', sparkle: '#ffc93c', anger: '#ff5d73', ledge: '#f3c9a8', ledgeTop: '#e8b38c', paw: '#fff7f0', pawLine: '#6b3f3f', heart: '#ff6f91' },
    openEye(ctx, e, p, side) {
      ctx.beginPath(); ctx.ellipse(e.cx, e.cy, e.w / 2, e.h / 2, 0, 0, Math.PI * 2); ctx.fill();
      ctx.save(); ctx.fillStyle = '#fff';
      ctx.beginPath(); ctx.arc(e.cx - e.w * 0.16 + p.lookX * 0.15, e.cy - e.h * 0.18, e.w * 0.19, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(e.cx + e.w * 0.17 + p.lookX * 0.15, e.cy + e.h * 0.2, e.w * 0.08, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
    },
    draw(ctx, p, o) {
      const G = this.G, D = '#2b2024';
      ctx.fillStyle = '#fff1e6'; ctx.fillRect(0, 0, 320, 240);
      ctx.translate(0, Math.sin(o.t / 1300) * 2);
      drawBlush(ctx, G, p, '#ffb3c1', true);
      ctx.save(); ctx.fillStyle = D; ctx.strokeStyle = D; ctx.lineWidth = 9; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      drawEye(ctx, this, G, p, o, -1); drawEye(ctx, this, G, p, o, 1);
      ctx.strokeStyle = '#6b3f3f'; ctx.lineWidth = 4.5;
      if (p.mouth !== 'none') drawMouth(ctx, p.mouth, 160, G.cy + p.eyeY + 58, p.mouth === 'w' ? 9 : 13, '#ff8fa3');
      ctx.restore();
      drawExtras(ctx, G, p, o, this.c);
      drawLedge(ctx, p, o, this.c);
    },
  },
  cat: {
    name: '猫猫脸', desc: '圆耳朵、大大的玻璃珠眼睛、白嘴筒和额头小花纹',
    note: '<b>优点</b>：一眼就是“宠物”，耳朵会跟着情绪竖起或耷拉、偶尔抖一下，瞳孔会随好奇放大。<br><b>注意</b>：渐变和细节多，ESP32 上要把背景和耳朵预先画成图片再叠加；如果以后外壳本身做了实体猫耳，屏幕上的耳朵可以去掉。',
    G: { gap: 62, cy: 122, w: 66, h: 68 },
    c: { extra: '#a78bdf', q: '#f08a3c', tear: '#6fa8ff', sparkle: '#ffc93c', anger: '#ff5d73', ledge: '#e9a96b', ledgeTop: '#d9925a',
         paw: '#fff3e3', pawLine: '#c98a5a', heart: '#ff6f91',
         pos: { z: [178, 70], q: [270, 166], dots: [196, 52], anger: [226, 44] } },
    openEye(ctx, e, p) {
      const r = Math.min(e.w, e.h) / 2;
      ctx.save();
      // 虹膜：上深下浅，像一颗玻璃珠
      const g = ctx.createLinearGradient(0, e.cy - e.h / 2, 0, e.cy + e.h / 2);
      g.addColorStop(0, '#5aa848'); g.addColorStop(1, '#c8ee8e');
      ctx.fillStyle = g; ctx.beginPath(); ctx.ellipse(e.cx, e.cy, e.w / 2, e.h / 2, 0, 0, Math.PI * 2); ctx.fill();
      // 圆圆的小猫瞳，好奇、倾听时放大，犯困时缩小
      const pr = r * (0.42 + p.pupil * 0.38);
      ctx.fillStyle = '#231a16'; ctx.beginPath(); ctx.ellipse(e.cx + p.lookX * 0.15, e.cy + 2, pr * 0.92, pr, 0, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.beginPath(); ctx.arc(e.cx - r * 0.3, e.cy - r * 0.32, r * 0.27, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(e.cx + r * 0.3, e.cy + r * 0.3, r * 0.11, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = '#4a3426'; ctx.lineWidth = 3.5; ctx.beginPath(); ctx.ellipse(e.cx, e.cy, e.w / 2, e.h / 2, 0, 0, Math.PI * 2); ctx.stroke();
      ctx.restore();
    },
    // 圆角三角形：耳朵尖和鼻子都用它，避免尖角显得凶
    roundTri(ctx, a, tip, b, rad) {
      const near = (q, d) => { const len = Math.hypot(q[0] - tip[0], q[1] - tip[1]); return [tip[0] + (q[0] - tip[0]) * d / len, tip[1] + (q[1] - tip[1]) * d / len]; };
      const p1 = near(a, rad), p2 = near(b, rad);
      ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...p1); ctx.quadraticCurveTo(...tip, ...p2); ctx.lineTo(...b); ctx.closePath();
    },
    ear(ctx, p, side, t) {
      const droop = Math.max(0, -p.ear), perk = Math.max(0, p.ear);
      const ph = t % 5200, twitch = side > 0 && ph < 260 ? Math.sin(ph / 260 * Math.PI) * 10 : 0;  // 右耳偶尔抖一下
      const tip = [160 + side * (108 + droop * 38), 14 + droop * 30 - perk * 10 - twitch];
      const b1 = [160 + side * 50, 64], b2 = [160 + side * 152, 96];
      ctx.fillStyle = '#f2ac68'; this.roundTri(ctx, b1, tip, b2, 18); ctx.fill();
      const cx = (b1[0] + b2[0] + tip[0]) / 3, cy = (b1[1] + b2[1] + tip[1]) / 3, k = 0.56;
      const sh = q => [cx + (q[0] - cx) * k, cy + (q[1] - cy) * k + 5];
      ctx.fillStyle = '#ffc2cf'; this.roundTri(ctx, sh(b1), sh(tip), sh(b2), 10); ctx.fill();
      // 耳朵里的两撮白毛
      ctx.strokeStyle = '#fff3e3'; ctx.lineWidth = 2.5; ctx.lineCap = 'round';
      const base = sh([(b1[0] + b2[0]) / 2, (b1[1] + b2[1]) / 2]);
      for (const d of [-6, 6]) { ctx.beginPath(); ctx.moveTo(base[0] + d, base[1] - 2); ctx.lineTo(base[0] + d * 0.4 + (tip[0] - base[0]) * 0.25, base[1] - 14); ctx.stroke(); }
    },
    draw(ctx, p, o) {
      const G = this.G, D = '#4a3426';
      const bg = ctx.createRadialGradient(160, 130, 30, 160, 130, 230);
      bg.addColorStop(0, '#fff2df'); bg.addColorStop(1, '#ffd9ae');
      ctx.fillStyle = bg; ctx.fillRect(0, 0, 320, 240);
      ctx.translate(0, Math.sin(o.t / 1300) * 2);
      this.ear(ctx, p, -1, o.t); this.ear(ctx, p, 1, o.t);
      // 额头三道小花纹
      ctx.save(); ctx.strokeStyle = '#f3b57c'; ctx.lineWidth = 6; ctx.lineCap = 'round';
      for (const i of [-1, 0, 1]) { ctx.beginPath(); ctx.moveTo(160 + i * 15, 30 + Math.abs(i) * 7); ctx.lineTo(160 + i * 13, 50 + Math.abs(i) * 3); ctx.stroke(); }
      ctx.restore();
      const ny = G.cy + p.eyeY * 0.6 + 46;
      if (!p.peek) { ctx.fillStyle = '#fffaf2'; ctx.beginPath(); ctx.ellipse(160, ny + 12, 46, 27, 0, 0, Math.PI * 2); ctx.fill(); }  // 白嘴筒
      drawBlush(ctx, G, { ...p, blush: Math.max(p.blush, 0.45) }, '#ffadc0', false);
      ctx.save(); ctx.fillStyle = D; ctx.strokeStyle = D; ctx.lineWidth = 8; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      drawEye(ctx, this, G, p, o, -1); drawEye(ctx, this, G, p, o, 1);
      if (!p.peek) {
        ctx.fillStyle = '#ff8fa8'; this.roundTri(ctx, [150, ny], [160, ny + 9], [170, ny], 4); ctx.fill();
        ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(156, ny + 2, 1.8, 0, Math.PI * 2); ctx.fill();
        ctx.lineWidth = 3.5;
        if (p.mouth === 'o') drawMouth(ctx, 'o', 160, ny + 18, 10, '#ff7f98');
        else if (p.mouth === 'frown') drawMouth(ctx, 'frown', 160, ny + 18, 9);
        else {
          drawMouth(ctx, 'w', 160, ny + 12, 7);
          if (p.mouth === 'smile') { ctx.fillStyle = '#ff7f98'; ctx.beginPath(); ctx.ellipse(160, ny + 23, 6, 6, 0, 0, Math.PI); ctx.fill(); }  // 开心时吐一点小舌头
        }
        ctx.strokeStyle = '#d6a27c'; ctx.lineWidth = 2.2;
        for (const side of [-1, 1]) for (const i of [0, 1]) {
          ctx.beginPath(); ctx.moveTo(160 + side * 48, ny + 6 + i * 10);
          ctx.quadraticCurveTo(160 + side * 72, ny + 1 + i * 10, 160 + side * 96, ny - 5 + i * 16); ctx.stroke();
        }
      }
      ctx.restore();
      drawExtras(ctx, G, p, o, this.c);
      drawLedge(ctx, p, o, this.c);
    },
  },
  bajo: {
    name: '呆猫（八条风）', desc: '银渐层小灰猫：粗黑描边、超大黑眼睛、表情配大字',
    note: '<b>优点</b>：呆萌感和表情包感最强，配上大字后情绪一眼就懂；黑眼睛大到占半张脸，摸它时的反差萌很足。<br><b>注意</b>：这是参考“呆猫八条”画风自己画的猫，不是原图；自己玩没问题，如果以后要做成商品卖，需要原作者授权。',
    G: { gap: 64, cy: 140, w: 84, h: 84 },
    c: { extra: '#5a5f6b', tear: '#7fb8ff', sparkle: '#f5c518', anger: '#ff5d73', ledge: '#ece8df', ledgeTop: '#141414',
         paw: '#ffffff', pawLine: '#141414', heart: '#ff6f91', pos: { dots: [214, 40], anger: [262, 74] } },
    captions: { listen: '嗯？', curious: '嗯？？', pout: '哼！', sad: '呜呜…', lonely: '在吗…', sleepy: '好困', sleep: 'Zzz',
                happy: '嘿嘿', shy: '害羞', peek: '偷看', think: '让我想想' },
    openEye(ctx, e, p) {
      const R = Math.min(e.w, e.h) / 2;
      const dot = (x, y, r) => { ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); };
      ctx.save();
      ctx.fillStyle = '#141414'; dot(e.cx, e.cy, R);
      ctx.fillStyle = '#e4e98f'; dot(e.cx, e.cy, R - 4);               // 一圈细细的黄绿色虹膜
      ctx.fillStyle = '#141414'; dot(e.cx + p.lookX * 0.12, e.cy + p.lookY * 0.12, (R - 4) * (0.86 + p.pupil * 0.08));
      ctx.fillStyle = '#fff'; dot(e.cx - R * 0.32, e.cy - R * 0.32, R * 0.26); dot(e.cx + R * 0.28, e.cy + R * 0.27, R * 0.11);
      ctx.restore();
    },
    head(ctx, p, t) {
      const K = '#141414', droop = Math.max(0, -p.ear), perk = Math.max(0, p.ear);
      const ph = t % 6100, twitch = ph < 240 ? Math.sin(ph / 240 * Math.PI) * 8 : 0;   // 左耳偶尔抖一下
      ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      for (const side of [-1, 1]) {   // 耳朵先画，耳根会被头盖住
        const tip = [160 + side * (100 + droop * 30), 16 + droop * 26 - perk * 8 - (side < 0 ? twitch : 0)];
        const b1 = [160 + side * 40, 60], b2 = [160 + side * 118, 100];
        ctx.fillStyle = '#b4b6bf'; STYLES.cat.roundTri(ctx, b1, tip, b2, 14); ctx.fill();
        ctx.strokeStyle = K; ctx.lineWidth = 5; ctx.stroke();
        const cx = (b1[0] + b2[0] + tip[0]) / 3, cy = (b1[1] + b2[1] + tip[1]) / 3, k = 0.5;
        const sh = q => [cx + (q[0] - cx) * k, cy + (q[1] - cy) * k + 4];
        ctx.fillStyle = '#f6b8c8'; STYLES.cat.roundTri(ctx, sh(b1), sh(tip), sh(b2), 8); ctx.fill();
      }
      // 头是一个大圆团，下半截超出屏幕；灰色头顶渐变到白色下巴，就是银渐层
      const shape = () => { ctx.beginPath(); ctx.ellipse(160, 196, 150, 150, 0, 0, Math.PI * 2); };
      const g = ctx.createLinearGradient(0, 46, 0, 240);
      g.addColorStop(0, '#aeb0b9'); g.addColorStop(0.45, '#d5d6db'); g.addColorStop(0.75, '#f4f4f6'); g.addColorStop(1, '#ffffff');
      shape(); ctx.fillStyle = g; ctx.fill();
      ctx.save(); shape(); ctx.clip(); ctx.fillStyle = '#ffffff'; ctx.beginPath(); ctx.ellipse(160, 216, 92, 58, 0, 0, Math.PI * 2); ctx.fill(); ctx.restore();
      shape(); ctx.strokeStyle = K; ctx.lineWidth = 5; ctx.stroke();
    },
    draw(ctx, p, o) {
      const G = this.G, K = '#141414';
      ctx.fillStyle = '#f7f6f2'; ctx.fillRect(0, 0, 320, 240);
      ctx.translate(0, Math.sin(o.t / 1300) * 2);
      ctx.save(); ctx.translate(0, p.peek ? p.eyeY : p.eyeY * 0.4); this.head(ctx, p, o.t); ctx.restore();
      drawBlush(ctx, { ...G, gap: G.gap + 24, h: G.h * 0.7 }, { ...p, blush: Math.max(p.blush, 0.3) }, '#ffb6c8', false);
      ctx.save(); ctx.fillStyle = K; ctx.strokeStyle = K; ctx.lineWidth = 7; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      drawEye(ctx, this, G, p, o, -1); drawEye(ctx, this, G, p, o, 1);
      // 半闭眼时补一道粗眼皮线，“呆”和“无语”的神态全靠它
      if (p.mode === 'open' && o.blink < 0.8) for (const side of [-1, 1]) {
        const e = eyeBox(G, p, side), lt = Math.max(p.lidT, o.blink);
        if (lt < 0.04 && Math.abs(p.brow) < 0.04) continue;
        const x0 = e.cx - e.w / 2, x1 = e.cx + e.w / 2;
        ctx.save(); ctx.beginPath(); ctx.arc(e.cx, e.cy, e.w / 2 + 1, 0, Math.PI * 2); ctx.clip();
        ctx.lineWidth = 6; ctx.beginPath(); ctx.moveTo(x0, lidY(e, p, side, lt, x0)); ctx.lineTo(x1, lidY(e, p, side, lt, x1)); ctx.stroke();
        ctx.restore();
      }
      const my = G.cy + p.eyeY + 56;
      if (!p.peek) {
        ctx.lineWidth = 3.5;
        if (p.mouth === 'o') drawMouth(ctx, 'o', 160, my, 9, '#ff8fa3');
        else if (p.mouth === 'frown') drawMouth(ctx, 'frown', 160, my, 8);
        else if (p.mouth === 'flat') drawMouth(ctx, 'flat', 160, my + 2, 7);
        else {
          drawMouth(ctx, 'w', 160, my, 6);
          if (p.mouth === 'smile') { ctx.fillStyle = '#ff8fa3'; ctx.beginPath(); ctx.ellipse(160, my + 9, 5, 5, 0, 0, Math.PI); ctx.fill(); }
        }
      }
      ctx.restore();
      // 问号、Z 这些改成表情包式的大字，其余小特效照常
      drawExtras(ctx, G, { ...p, extra: ['q', '!', 'z', 'zzz'].includes(p.extra) ? null : p.extra }, o, this.c);
      drawLedge(ctx, p, o, this.c);
      const cap = this.captions[o.face];
      if (cap && !(o.face === 'peek' && o.age < 1000)) {
        ctx.save(); ctx.font = '900 30px "Microsoft YaHei", "PingFang SC", sans-serif'; ctx.textAlign = 'center'; ctx.lineJoin = 'round';
        const y = 40 + Math.sin(o.t / 400) * 2;
        ctx.lineWidth = 7; ctx.strokeStyle = '#f7f6f2'; ctx.strokeText(cap, 160, y);
        ctx.fillStyle = K; ctx.fillText(cap, 160, y);
        ctx.restore();
      }
    },
  },
  pixel: {
    name: '像素复古', desc: '老式电子宠物的单色像素风（就是第一阶段假设备那套）',
    note: '<b>优点</b>：最有复古电子宠物的味道，ESP32 上画起来最轻松、最省电。<br><b>注意</b>：表情细腻度最低，彩屏的优势用不上。',
    draw(ctx, p, o) {
      ctx.fillStyle = '#b9c79a'; ctx.fillRect(0, 0, 320, 240);
      pix.render(o);
      ctx.imageSmoothingEnabled = false;
      ctx.globalAlpha = 0.12; ctx.drawImage(pix.cv, 2.5, 42.5, 320, 160); ctx.globalAlpha = 1;  // 液晶的淡淡残影
      ctx.drawImage(pix.cv, 0, 40, 320, 160);
      ctx.fillStyle = '#26301a';
      ctx.fillRect(272, 12, 30, 14); ctx.fillStyle = '#b9c79a'; ctx.fillRect(275, 15, 8, 8); ctx.fillStyle = '#26301a'; ctx.fillRect(302, 16, 3, 6);
      for (let i = 0; i < 3; i++) ctx.fillRect(16 + i * 7, 24 - i * 5, 5, 4 + i * 5);
    },
  },
};

// 像素风：沿用第一阶段 128×64 的画法，再放大 2.5 倍
const pix = (() => {
  const cv = document.createElement('canvas'); cv.width = 128; cv.height = 64;
  const x = cv.getContext('2d'); const ON = '#26301a';
  const px = (a, b, w = 1, h = 1) => x.fillRect(Math.round(a), Math.round(b), w, h);
  const clr = (a, b, w = 1, h = 1) => x.clearRect(Math.round(a), Math.round(b), w, h);
  const GL = { z: ['1111', '0010', '0100', '1111'], Z: ['11111', '00010', '00100', '01000', '11111'], '?': ['0110', '1001', '0010', '0100', '0000', '0100'], '!': ['1', '1', '1', '0', '1'] };
  const glyph = (ch, a, b, s = 1) => (GL[ch] || []).forEach((row, j) => [...row].forEach((v, i) => { if (v === '1') px(a + i * s, b + j * s, s, s); }));
  const eye = {
    open(cx, cy, h = 16) { px(cx - 6, cy - h / 2, 12, h); clr(cx - 6, cy - h / 2); clr(cx + 5, cy - h / 2); clr(cx - 6, cy + h / 2 - 1); clr(cx + 5, cy + h / 2 - 1); clr(cx + 1, cy - h / 2 + 2, 3, 3); },
    closed(cx, cy) { px(cx - 6, cy, 12, 2); },
    happy(cx, cy) { for (let i = -6; i <= 5; i++) px(cx + i, cy - 4 + Math.abs(i + 0.5) * 0.8, 1, 2); },
    half(cx, cy) { px(cx - 6, cy - 1, 12, 2); px(cx - 5, cy + 1, 10, 4); },
    look(cx, cy, dx, dy = 0) { px(cx - 6, cy - 8, 12, 16); clr(cx - 2 + dx, cy - 3 + dy, 4, 6); },
    pout(cx, cy, dir) { for (let i = 0; i < 6; i++) { px(cx + dir * (i - 3), cy - 5 + i, 2, 2); px(cx + dir * (i - 3), cy + 5 - i, 2, 2); } },
  };
  const L = 44, R = 84, EY = 28;
  const brows = () => { for (let i = 0; i < 8; i++) { const d = i * 0.4; px(L - 4 + i, EY - 10 - d, 1, 1); px(R + 3 - i, EY - 10 - d, 1, 1); } };
  const mouth = {
    w(y = 44) { px(58, y, 2, 1); px(60, y + 1, 2, 1); px(62, y, 3, 1); px(65, y + 1, 2, 1); px(67, y, 2, 1); },
    flat(y = 45) { px(60, y, 8, 1); },
    o(y = 43) { px(62, y, 4, 1); px(61, y + 1, 1, 2); px(66, y + 1, 1, 2); px(62, y + 3, 4, 1); },
    frown(y = 46) { px(59, y + 1, 2, 1); px(61, y, 6, 1); px(67, y + 1, 2, 1); },
  };
  const blush = () => { for (let i = 0; i < 3; i++) { px(27 + i * 3, 40, 2, 1); px(93 + i * 3, 40, 2, 1); } };
  const ears = () => { for (let r = 0; r < 9; r++) { px(30 - r / 2, 2 + r, r + 1, 1); px(98 - r / 2, 2 + r, r + 1, 1); } };
  const F = {
    idle(t, b) { if (b) { eye.closed(L, EY); eye.closed(R, EY); } else { eye.open(L, EY); eye.open(R, EY); } mouth.w(); },
    listen() { ears(); eye.open(L, EY + 2, 18); eye.open(R, EY + 2, 18); mouth.o(46); },
    think(t) { const dx = Math.floor(t / 500) % 2 ? 2 : -2; eye.look(L, EY, dx, -2); eye.look(R, EY, dx, -2); mouth.flat(); const n = Math.floor(t / 300) % 4; for (let i = 0; i < n; i++) px(100 + i * 4, 8, 2, 2); },
    happy() { eye.happy(L, EY); eye.happy(R, EY); mouth.w(42); blush(); },
    shy(t, b) { if (b) { eye.closed(L, EY + 2); eye.closed(R, EY + 2); } else { eye.look(L, EY + 2, 0, 3); eye.look(R, EY + 2, 0, 3); } mouth.flat(46); blush(); },
    pout() { eye.pout(L, EY, 1); eye.pout(R, EY, -1); mouth.frown(); },
    sad(t, b) { brows(); if (b) { eye.closed(L, EY); eye.closed(R, EY); } else { eye.look(L, EY, 0, 3); eye.look(R, EY, 0, 3); } mouth.frown(); if (Math.floor(t / 800) % 2) px(L - 5, EY + 10, 2, 3); },
    lonely(t, b) { brows(); if (b) { eye.closed(L, EY + 2); eye.closed(R, EY + 2); } else { eye.look(L, EY + 2, -2, 4); eye.look(R, EY + 2, -2, 4); } mouth.flat(47); },
    curious(t, b) { if (b) eye.closed(L, EY); else eye.open(L, EY, 18); eye.open(R, EY, 12); mouth.o(); glyph('?', 104, 4 + (Math.floor(t / 400) % 2), 2); },
    sleepy(t) { eye.half(L, EY); eye.half(R, EY); mouth.flat(); glyph('z', 104, 10 - (Math.floor(t / 600) % 3)); },
    sleep(t) { eye.closed(L, EY + 2); eye.closed(R, EY + 2); mouth.flat(45); const k = Math.floor(t / 700) % 3; glyph('z', 98, 18 - k * 3); if (k > 0) glyph('Z', 106, 8 - k * 2); },
    peek(t, b, age) {
      const rise = Math.min(1, age / 900), y = 62 - rise * 24;
      if (b) { eye.closed(L, y); eye.closed(R, y); } else { eye.look(L, y, 0, -3); eye.look(R, y, 0, -3); }
      clr(0, 50, 128, 14); px(0, 50, 128, 2);
      if (rise === 1) glyph('!', 108, 10, 2);
    },
  };
  return {
    cv,
    render(o) {
      x.clearRect(0, 0, 128, 64); x.fillStyle = ON;
      x.save(); x.translate(0, Math.sin(o.t / 1200) > 0 ? 1 : 0);
      (F[o.face] || F.idle)(o.t, o.blink > 0.3, o.age);
      x.restore();
    },
  };
})();

