module util;

import graph;

public struct ScreenOutput {
	public f32x4 position : SV_Position;
	public f32x2 uv : UV;
}

[shader("vertex")]
ScreenOutput screen(u32 vertex: SV_VertexID) {
	let uv = f32x2((vertex << 1) & 2, vertex & 2);
	let position = f32x4(uv * 2.0f - 1.0f, 0.0f, 1.0f);
	ScreenOutput ret = { position, uv };
	return ret;
}

// https://www.shadertoy.com/view/4sBSWW/
f32 digit_bin(const i32 x) {
	return x == 0	? 480599.0
		   : x == 1 ? 139810.0
		   : x == 2 ? 476951.0
		   : x == 3 ? 476999.0
		   : x == 4 ? 350020.0
		   : x == 5 ? 464711.0
		   : x == 6 ? 464727.0
		   : x == 7 ? 476228.0
		   : x == 8 ? 481111.0
		   : x == 9 ? 481095.0
					: 0.0;
}

public f32 print(f32x2 vStringCoords, f32 fValue, f32 fMaxDigits, f32 fDecimalPlaces) {
	if ((vStringCoords.y < 0.0) || (vStringCoords.y >= 1.0))
		return 0.0;

	let bNeg = (fValue < 0.0);
	fValue = abs(fValue);

	let fLog10Value = log2(abs(fValue)) / log2(10.0);
	let fBiggestIndex = max(floor(fLog10Value), 0.0);
	var fDigitIndex = fMaxDigits - floor(vStringCoords.x);
	var fCharBin = 0.0;
	if (fDigitIndex > (-fDecimalPlaces - 1.01)) {
		if (fDigitIndex > fBiggestIndex) {
			if ((bNeg) && (fDigitIndex < (fBiggestIndex + 1.5)))
				fCharBin = 1792.0;
		} else {
			if (fDigitIndex == -1.0) {
				if (fDecimalPlaces > 0.0)
					fCharBin = 2.0;
			} else {
				var fReducedRangeValue = fValue;
				if (fDigitIndex < 0.0) {
					fReducedRangeValue = fract(fValue);
					fDigitIndex += 1.0;
				}
				let fDigitValue = (abs(fReducedRangeValue / (pow(10.0, fDigitIndex))));
				fCharBin = digit_bin(int(floor(fmod(fDigitValue, 10.0))));
			}
		}
	}
	return floor(
		fmod((fCharBin / pow(2.0, floor(fract(vStringCoords.x) * 4.0) + (floor(vStringCoords.y * 5.0) * 4.0))), 2.0));
}

public f32 print(f32x2 uv, f32x2 start_uv, f32x2 vFontSize, f32 fValue, f32 fMaxDigits, f32 fDecimalPlaces) {
	var vStringCharCoords = (uv - start_uv) / vFontSize;
	vStringCharCoords.y = 1.f - vStringCharCoords.y;
	return print(vStringCharCoords, fValue, fMaxDigits, fDecimalPlaces);
}
