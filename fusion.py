def fusion_prediction(branchA_result, branchB_result, clip_result):
    """
    Performs weighted fusion of Branch A, Branch B and CLIP.
    """

    # -----------------------------
    # Model Weights
    # -----------------------------

    weight_A = 0.35
    weight_B = 0.45
    weight_CLIP = 0.20

    # -----------------------------
    # Weighted Contributions
    # -----------------------------

    scoreA = branchA_result["confidence"] * weight_A
    scoreB = branchB_result["confidence"] * weight_B
    scoreCLIP = clip_result["confidence"] * weight_CLIP

    fake_score = 0
    real_score = 0

    if branchA_result["prediction"] == "Fake":
        fake_score += scoreA
    else:
        real_score += scoreA

    if branchB_result["prediction"] == "Fake":
        fake_score += scoreB
    else:
        real_score += scoreB

    if clip_result["prediction"] == "Fake":
        fake_score += scoreCLIP
    else:
        real_score += scoreCLIP

    if fake_score > real_score:
        final_prediction = "Fake"
        final_confidence = fake_score
    else:
        final_prediction = "Real"
        final_confidence = real_score

    return {
        "prediction": final_prediction,
        "confidence": final_confidence,
        "fake_score": fake_score,
        "real_score": real_score
    }


def generation_method(branchA_result, branchB_result, fusion_result):
    """
    Estimates the most likely fake generation method.
    """

    if fusion_result["prediction"] == "Real":

        return {
            "method": "Authentic Image",
            "reliability": "N/A",
            "reason": "Fusion engine classified the image as authentic."
        }

    gan_conf = branchA_result["confidence"]
    diff_conf = branchB_result["confidence"]

    difference = abs(gan_conf - diff_conf)

    if gan_conf > diff_conf:

        method = "Likely GAN-generated"

        reason = (
            f"GAN detector confidence ({gan_conf*100:.2f}%) "
            f"is higher than Diffusion detector confidence "
            f"({diff_conf*100:.2f}%)."
        )

    else:

        method = "Likely Diffusion-generated"

        reason = (
            f"Diffusion detector confidence ({diff_conf*100:.2f}%) "
            f"is higher than GAN detector confidence "
            f"({gan_conf*100:.2f}%)."
        )

    if difference >= 0.20:
        reliability = "Very High"

    elif difference >= 0.10:
        reliability = "High"

    elif difference >= 0.05:
        reliability = "Moderate"

    else:
        reliability = "Low"

    return {
        "method": method,
        "reliability": reliability,
        "reason": reason
    }